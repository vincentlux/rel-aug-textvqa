# Copyright (c) Facebook, Inc. and its affiliates.

"""
The processors exist in MMF to make data processing pipelines in various
datasets as similar as possible while allowing code reuse.

The processors also help maintain proper abstractions to keep only what matters
inside the dataset's code. This allows us to keep the dataset ``__getitem__``
logic really clean and no need about maintaining opinions about data type.
Processors can work on both images and text due to their generic structure.

To create a new processor, follow these steps:

1. Inherit the ``BaseProcessor`` class.
2. Implement ``_call`` function which takes in a dict and returns a dict with
   same keys preprocessed as well as any extra keys that need to be returned.
3. Register the processor using ``@registry.register_processor('name')`` to
   registry where 'name' will be used to refer to your processor later.

In processor's config you can specify ``preprocessor`` option to specify
different kind of preprocessors you want in your dataset.

Let's break down processor's config inside a dataset (VQA2.0) a bit to understand
different moving parts.

Config::

    dataset_config:
      vqa2:
        data_dir: ${env.data_dir}
        processors:
          text_processor:
            type: vocab
            params:
              max_length: 14
              vocab:
                type: intersected
                embedding_name: glove.6B.300d
                vocab_file: vqa2/defaults/extras/vocabs/vocabulary_100k.txt
              preprocessor:
                type: simple_sentence
                params: {}

``BaseDataset`` will init the processors and they will available inside your
dataset with same attribute name as the key name, for e.g. `text_processor` will
be available as `self.text_processor` inside your dataset. As is with every module
in MMF, processor also accept a ``DictConfig`` with a `type` and `params`
attributes. `params` defined the custom parameters for each of the processors.
By default, processor initialization process will also init `preprocessor` attribute
which can be a processor config in itself. `preprocessor` can be then be accessed
inside the processor's functions.

Example::

    from mmf.common.registry import registry
    from mmf.datasets.processors import BaseProcessor

    @registry.register_processor('my_processor')
    class MyProcessor(BaseProcessor):
        def __init__(self, config, *args, **kwargs):
            return

        def __call__(self, item, *args, **kwargs):
            text = item['text']
            text = [t.strip() for t in text.split(" ")]
            return {"text": text}
"""

import collections
import copy
import logging
import os
import random
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Union

import numpy as np
import torch
from src.common.registry import registry
from src.common.typings import ProcessorConfigType
from src.utils.configuration import get_mmf_cache_dir, get_mmf_env
from src.utils.distributed import is_master, synchronize
from src.utils.file_io import PathManager
from src.utils.text import VocabDict
from src.utils.vocab import Vocab, WordToVectorDict


logger = logging.getLogger(__name__)


class Processor:
    """Wrapper class used by MMF to initialized processor based on their
    ``type`` as passed in configuration. It retrieves the processor class
    registered in registry corresponding to the ``type`` key and initializes
    with ``params`` passed in configuration. All functions and attributes of
    the processor initialized are directly available via this class.

    Args:
        config (DictConfig): DictConfig containing ``type`` of the processor to
                             be initialized and ``params`` of that processor.

    """

    def __init__(self, config: ProcessorConfigType, *args, **kwargs):
        if not hasattr(config, "type"):
            raise AttributeError(
                "Config must have 'type' attribute to specify type of processor"
            )
        print(config.type)
        processor_class = registry.get_processor_class(config.type)
        print(processor_class)
        params = {}
        if "params" not in config:
            logger.warning(
                "Config doesn't have 'params' attribute to "
                "specify parameters of the processor "
                f"of type {config.type}. Setting to default {{}}"
            )
        else:
            params = config.params

        self.processor = processor_class(params, *args, **kwargs)

        self._dir_representation = dir(self)

    def __call__(self, item, *args, **kwargs):
        return self.processor(item, *args, **kwargs)

    def __getattr__(self, name):
        if "_dir_representation" in self.__dict__ and name in self._dir_representation:
            return getattr(self, name)
        elif "processor" in self.__dict__ and hasattr(self.processor, name):
            return getattr(self.processor, name)
        else:
            raise AttributeError(f"The processor {name} doesn't exist in the registry.")



class BaseProcessor:
    """Every processor in MMF needs to inherit this class for compatibility
    with MMF. End user mainly needs to implement ``__call__`` function.

    Args:
        config (DictConfig): Config for this processor, containing `type` and
                             `params` attributes if available.

    """

    def __init__(self, config: Dict[str, Any], *args, **kwargs):
        return

    def __call__(self, item: Any, *args, **kwargs) -> Any:
        """Main function of the processor. Takes in a dict and returns back
        a dict

        Args:
            item (Dict): Some item that needs to be processed.

        Returns:
            Dict: Processed dict.

        """
        return item


@registry.register_processor("simple_word")
class SimpleWordProcessor(BaseProcessor):
    """Tokenizes a word and processes it.

    Attributes:
        tokenizer (function): Type of tokenizer to be used.

    """

    def __init__(self, *args, **kwargs):
        from src.utils.text import word_tokenize

        self.tokenizer = word_tokenize

    def __call__(self, item, *args, **kwargs):
        return {"text": self.tokenizer(item["text"], *args, **kwargs)}


@registry.register_processor("vocab")
class VocabProcessor(BaseProcessor):
    """Use VocabProcessor when you have vocab file and you want to process
    words to indices. Expects UNK token as "<unk>" and pads sentences using
    "<pad>" token. Config parameters can have ``preprocessor`` property which
    is used to preprocess the item passed and ``max_length`` property which
    points to maximum length of the sentence/tokens which can be convert to
    indices. If the length is smaller, the sentence will be padded. Parameters
    for "vocab" are necessary to be passed.

    **Key**: vocab

    Example Config::

        dataset_config:
          vqa2:
            data_dir: ${env.data_dir}
            processors:
              text_processor:
                type: vocab
                params:
                  max_length: 14
                  vocab:
                    type: intersected
                    embedding_name: glove.6B.300d
                    vocab_file: vqa2/defaults/extras/vocabs/vocabulary_100k.txt

    Args:
        config (DictConfig): node containing configuration parameters of
                             the processor

    Attributes:
        vocab (Vocab): Vocab class object which is abstraction over the vocab
                       file passed.
    """

    MAX_LENGTH_DEFAULT = 50
    PAD_TOKEN = "<pad>"
    PAD_INDEX = 0

    def __init__(self, config, *args, **kwargs):
        if not hasattr(config, "vocab"):
            raise AttributeError(
                "config passed to the processor has no attribute vocab"
            )

        self.vocab = Vocab(*args, **config.vocab, **kwargs)
        self._init_extras(config)

    def _init_extras(self, config, *args, **kwargs):
        self.preprocessor = None

        if hasattr(config, "max_length"):
            self.max_length = config.max_length
        else:
            warnings.warn(
                "No 'max_length' parameter in Processor's "
                "configuration. Setting to {}.".format(self.MAX_LENGTH_DEFAULT)
            )
            self.max_length = self.MAX_LENGTH_DEFAULT

        if "preprocessor" in config:
            self.preprocessor = Processor(config.preprocessor, *args, **kwargs)

            if self.preprocessor is None:
                raise ValueError(
                    f"No text processor named {config.preprocessor} is defined."
                )

    def __call__(self, item):
        """Call requires item to have either "tokens" attribute or either
        "text" attribute. If "text" is present, it will tokenized using
        the preprocessor.

        Args:
            item (Dict): Dict containing the "text" or "tokens".

        Returns:
            Dict: Dict containing indices in "text" key, "tokens" in "tokens"
                  key and "length" of the string in "length" key.

        """
        indices = None
        if not isinstance(item, dict):
            raise TypeError(
                "Argument passed to the processor must be "
                "a dict with either 'text' or 'tokens' as "
                "keys"
            )
        if "tokens" in item:
            tokens = item["tokens"]
            indices = self._map_strings_to_indices(item["tokens"])
        elif "text" in item:
            if self.preprocessor is None:
                raise AssertionError(
                    "If tokens are not provided, a text "
                    "processor must be defined in the config"
                )

            tokens = self.preprocessor({"text": item["text"]})["text"]
            indices = self._map_strings_to_indices(tokens)
        else:
            raise AssertionError(
                "A dict with either 'text' or 'tokens' keys "
                "must be passed to the processor"
            )

        tokens, length = self._pad_tokens(tokens)

        return {"text": indices, "tokens": tokens, "length": length}

    def _pad_tokens(self, tokens):
        padded_tokens = [self.PAD_TOKEN] * self.max_length
        token_length = min(len(tokens), self.max_length)
        padded_tokens[:token_length] = tokens[:token_length]
        token_length = torch.tensor(token_length, dtype=torch.long)
        return padded_tokens, token_length

    def get_pad_index(self):
        """Get index of padding <pad> token in vocabulary.

        Returns:
            int: index of the padding token.

        """
        return self.vocab.get_pad_index()

    def get_vocab_size(self):
        """Get size of the vocabulary.

        Returns:
            int: size of the vocabulary.

        """
        return self.vocab.get_size()

    def _map_strings_to_indices(self, tokens):
        length = min(len(tokens), self.max_length)
        tokens = tokens[:length]

        output = torch.zeros(self.max_length, dtype=torch.long)
        output.fill_(self.vocab.get_pad_index())

        for idx, token in enumerate(tokens):
            output[idx] = self.vocab.stoi[token]

        return output


@registry.register_processor("fasttext")
class FastTextProcessor(VocabProcessor):
    """FastText processor, similar to GloVe processor but returns FastText vectors.

    Args:
        config (DictConfig): Configuration values for the processor.

    """

    def __init__(self, config, *args, **kwargs):
        self._init_extras(config)
        self.config = config
        self._download_initially = config.get("download_initially", True)
        self._already_downloaded = False
        self._already_loaded = False

        if self._download_initially:
            self._try_download()

    def _try_download(self):
        _is_master = is_master()

        if self._already_downloaded:
            return

        needs_download = False

        if not hasattr(self.config, "model_file"):
            if _is_master:
                warnings.warn(
                    "'model_file' key is required but missing "
                    "from FastTextProcessor's config."
                )
            needs_download = True

        model_file = self.config.model_file
        # If model_file is already an existing path don't join to cache dir
        if not PathManager.exists(model_file):
            model_file = os.path.join(get_mmf_cache_dir(), model_file)

        if not PathManager.exists(model_file):
            if _is_master:
                warnings.warn(f"No model file present at {model_file}.")
            needs_download = True

        if needs_download:
            logger.info("Downloading FastText bin")
            model_file = self._download_model()

        self.model_file = model_file
        self._already_downloaded = True
        synchronize()

    def _download_model(self):
        _is_master = is_master()

        model_file_path = os.path.join(get_mmf_cache_dir(), "wiki.en.bin")

        if not _is_master:
            return model_file_path

        if PathManager.exists(model_file_path):
            logger.info(f"Vectors already present at {model_file_path}.")
            return model_file_path

        import requests
        from tqdm import tqdm

        from src.common.constants import FASTTEXT_WIKI_URL

        PathManager.mkdirs(os.path.dirname(model_file_path))
        response = requests.get(FASTTEXT_WIKI_URL, stream=True)

        with PathManager.open(model_file_path, "wb") as f:
            pbar = tqdm(
                total=int(response.headers["Content-Length"]) / 4096,
                miniters=50,
                disable=not _is_master,
            )

            idx = 0
            for data in response.iter_content(chunk_size=4096):
                if data:
                    if idx % 50 == 0:
                        pbar.update(len(data))
                    f.write(data)
                    idx += 1

            pbar.close()

        logger.info(f"fastText bin downloaded at {model_file_path}.")

        return model_file_path

    def _load_fasttext_model(self, model_file):
        if self._already_loaded:
            return

        from fasttext import load_model

        logger.info(f"Loading fasttext model now from {model_file}")

        self.model = load_model(model_file)
        # String to Vector
        self.stov = WordToVectorDict(self.model)
        logger.info("Finished loading fasttext model")

        self._already_loaded = True

    def _map_strings_to_indices(self, tokens):
        length = min(len(tokens), self.max_length)
        tokens = tokens[:length]

        output = torch.full(
            (self.max_length, self.model.get_dimension()),
            fill_value=self.PAD_INDEX,
            dtype=torch.float,
        )

        for idx, token in enumerate(tokens):
            output[idx] = torch.from_numpy(self.stov[token])

        return output

    def __call__(self, item):
        self._load_fasttext_model(self.model_file)
        return super().__call__(item)


@registry.register_processor("m4c_answer")
class M4CAnswerProcessor(BaseProcessor):
    """
    Process a TextVQA answer for iterative decoding in M4C
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.answer_vocab = VocabDict(config.vocab_file, *args, **kwargs)
        self.PAD_IDX = self.answer_vocab.word2idx("<pad>")
        self.BOS_IDX = self.answer_vocab.word2idx("<s>")
        self.EOS_IDX = self.answer_vocab.word2idx("</s>")
        self.UNK_IDX = self.answer_vocab.UNK_INDEX

        # make sure PAD_IDX, BOS_IDX and PAD_IDX are valid (not <unk>)
        assert self.PAD_IDX != self.answer_vocab.UNK_INDEX
        assert self.BOS_IDX != self.answer_vocab.UNK_INDEX
        assert self.EOS_IDX != self.answer_vocab.UNK_INDEX
        assert self.PAD_IDX == 0

        self.answer_preprocessor = Processor(config.preprocessor)
        assert self.answer_preprocessor is not None

        self.num_answers = config.num_answers
        self.max_length = config.max_length
        self.max_copy_steps = config.max_copy_steps
        assert self.max_copy_steps >= 1

        self.match_answer_to_unk = False

    def tokenize(self, sentence):
        return sentence.split()

    def match_answer_to_vocab_ocr_seq(
        self, answer, vocab2idx_dict, ocr2inds_dict, max_match_num=20
    ):
        """
        Match an answer to a list of sequences of indices
        each index corresponds to either a fixed vocabulary or an OCR token
        (in the index address space, the OCR tokens are after the fixed vocab)
        """
        num_vocab = len(vocab2idx_dict)

        answer_words = self.tokenize(answer)
        answer_word_matches = []
        for word in answer_words:
            # match answer word to fixed vocabulary
            matched_inds = []
            if word in vocab2idx_dict:
                matched_inds.append(vocab2idx_dict.get(word))
            # match answer word to OCR
            # we put OCR after the fixed vocabulary in the answer index space
            # so add num_vocab offset to the OCR index
            matched_inds.extend([num_vocab + idx for idx in ocr2inds_dict[word]])
            if len(matched_inds) == 0:
                if self.match_answer_to_unk:
                    matched_inds.append(vocab2idx_dict.get("<unk>"))
                else:
                    return []
            answer_word_matches.append(matched_inds)

        # expand per-word matched indices into the list of matched sequences
        if len(answer_word_matches) == 0:
            return []
        idx_seq_list = [()]
        for matched_inds in answer_word_matches:
            idx_seq_list = [
                seq + (idx,) for seq in idx_seq_list for idx in matched_inds
            ]
            if len(idx_seq_list) > max_match_num:
                idx_seq_list = idx_seq_list[:max_match_num]

        return idx_seq_list

    def get_vocab_size(self):
        answer_vocab_nums = self.answer_vocab.num_vocab
        answer_vocab_nums += self.max_length

        return answer_vocab_nums

    def get_true_vocab_size(self):
        return self.answer_vocab.num_vocab

    def compute_answer_scores(self, answers):
        gt_answers = list(enumerate(answers))
        unique_answers = sorted(set(answers))
        unique_answer_scores = [0] * len(unique_answers)
        for idx, unique_answer in enumerate(unique_answers):
            accs = []
            for gt_answer in gt_answers:
                other_answers = [item for item in gt_answers if item != gt_answer]
                matching_answers = [
                    item for item in other_answers if item[1] == unique_answer
                ]
                acc = min(1, float(len(matching_answers)) / 3)
                accs.append(acc)
            unique_answer_scores[idx] = sum(accs) / len(accs)
        unique_answer2score = {
            a: s for a, s in zip(unique_answers, unique_answer_scores)
        }
        return unique_answer2score

    def __call__(self, item):
        answers = item["answers"]

        if not answers:
            return {
                "sampled_idx_seq": None,
                "train_prev_inds": torch.zeros(self.max_copy_steps, dtype=torch.long),
            }

        answers = [self.answer_preprocessor({"text": a})["text"] for a in answers]
        assert len(answers) == self.num_answers

        # Step 1: calculate the soft score of ground-truth answers
        unique_answer2score = self.compute_answer_scores(answers)

        # Step 2: fill the first step soft scores for tokens
        scores = torch.zeros(
            self.max_copy_steps, self.get_vocab_size(), dtype=torch.float
        )

        # match answers to fixed vocabularies and OCR tokens.
        ocr2inds_dict = defaultdict(list)
        for idx, token in enumerate(item["tokens"]):
            ocr2inds_dict[token].append(idx)
        answer_dec_inds = [
            self.match_answer_to_vocab_ocr_seq(
                a, self.answer_vocab.word2idx_dict, ocr2inds_dict
            )
            for a in answers
        ]

        # Collect all the valid decoding sequences for each answer.
        # This part (idx_seq_list) was pre-computed in imdb (instead of online)
        # to save time
        all_idx_seq_list = []
        for answer, idx_seq_list in zip(answers, answer_dec_inds):
            all_idx_seq_list.extend(idx_seq_list)
            # fill in the soft score for the first decoding step
            score = unique_answer2score[answer]
            for idx_seq in idx_seq_list:
                score_idx = idx_seq[0]
                # the scores for the decoding Step 0 will be the maximum
                # among all answers starting with that vocab
                # for example:
                # if "red apple" has score 0.7 and "red flag" has score 0.8
                # the score for "red" at Step 0 will be max(0.7, 0.8) = 0.8
                scores[0, score_idx] = max(scores[0, score_idx], score)

        # train_prev_inds is the previous prediction indices in auto-regressive
        # decoding
        train_prev_inds = torch.zeros(self.max_copy_steps, dtype=torch.long)
        # train_loss_mask records the decoding steps where losses are applied
        train_loss_mask = torch.zeros(self.max_copy_steps, dtype=torch.float)
        if len(all_idx_seq_list) > 0:
            # sample a random decoding answer sequence for teacher-forcing
            idx_seq = all_idx_seq_list[np.random.choice(len(all_idx_seq_list))]
            dec_step_num = min(1 + len(idx_seq), self.max_copy_steps)
            train_loss_mask[:dec_step_num] = 1.0

            train_prev_inds[0] = self.BOS_IDX
            for t in range(1, dec_step_num):
                train_prev_inds[t] = idx_seq[t - 1]
                score_idx = idx_seq[t] if t < len(idx_seq) else self.EOS_IDX
                scores[t, score_idx] = 1.0
        else:
            idx_seq = ()

        answer_info = {
            "answers": answers,
            "answers_scores": scores,
            "sampled_idx_seq": idx_seq,
            "train_prev_inds": train_prev_inds,
            "train_loss_mask": train_loss_mask,
        }
        return answer_info


@registry.register_processor("phoc")
class PhocProcessor(VocabProcessor):
    """
    Compute PHOC features from text tokens
    """

    def __init__(self, config, *args, **kwargs):
        from src.utils.phoc import build_phoc

        self._build_phoc = build_phoc
        self._init_extras(config)
        self.config = config

    def _map_strings_to_indices(self, tokens):
        length = min(len(tokens), self.max_length)
        tokens = tokens[:length]

        phoc_dim = 604
        output = torch.full(
            (self.max_length, phoc_dim), fill_value=self.PAD_INDEX, dtype=torch.float
        )

        for idx, token in enumerate(tokens):
            output[idx] = torch.from_numpy(self._build_phoc(token))

        return output


@registry.register_processor("copy")
class CopyProcessor(BaseProcessor):
    """
    Copy boxes from numpy array
    """

    def __init__(self, config, *args, **kwargs):
        self.max_length = config.max_length

    def __call__(self, item):
        blob = item["blob"]
        final_blob = np.zeros((self.max_length,) + blob.shape[1:], blob.dtype)
        final_blob[: len(blob)] = blob[: len(final_blob)]

        return {"blob": torch.from_numpy(final_blob)}


@registry.register_processor("bbox")
class BBoxProcessor(VocabProcessor):
    """Generates bboxes in proper format.
    Takes in a dict which contains "info" key which is a list of dicts
    containing following for each of the the bounding box

    Example bbox input::

        {
            "info": [
                {
                    "bounding_box": {
                        "top_left_x": 100,
                        "top_left_y": 100,
                        "width": 200,
                        "height": 300
                    }
                },
                ...
            ]
        }


    This will further return a Sample in a dict with key "bbox" with last
    dimension of 4 corresponding to "xyxy". So sample will look like following:

    Example Sample::

        Sample({
            "coordinates": torch.Size(n, 4),
            "width": List[number], # size n
            "height": List[number], # size n
            "bbox_types": List[str] # size n, either xyxy or xywh.
            # currently only supports xyxy.
        })

    """

    def __init__(self, config, *args, **kwargs):
        from src.utils.dataset import build_bbox_tensors

        self.lambda_fn = build_bbox_tensors
        self._init_extras(config)

    def __call__(self, item):
        info = item["info"]
        if self.preprocessor is not None:
            info = self.preprocessor(info)

        return {"bbox": self.lambda_fn(info, self.max_length)}

