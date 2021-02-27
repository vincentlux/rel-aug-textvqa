# Copyright (c) Facebook, Inc. and its affiliates.
"""
Text utils module contains implementations for various decoding strategies like
Greedy, Beam Search and Nucleus Sampling.

In your model's config you can specify ``inference`` attribute to use these strategies
in the following way:

.. code::

   model_config:
       some_model:
           inference:
               - type: greedy
               - params: {}
"""
import os
import re
from collections import Counter
from itertools import chain

import torch
from src.common.registry import registry
from src.utils.file_io import PathManager
from src.utils.general import get_absolute_path


SENTENCE_SPLIT_REGEX = re.compile(r"(\W+)")

def tokenize(sentence, regex=SENTENCE_SPLIT_REGEX, keep=None, remove=None):
    if keep is None:
        keep = ["'s"]
    if remove is None:
        remove = [",", "?"]
    sentence = sentence.lower()

    for token in keep:
        sentence = sentence.replace(token, " " + token)

    for token in remove:
        sentence = sentence.replace(token, "")

    tokens = regex.split(sentence)
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens


def word_tokenize(word, remove=None):
    if remove is None:
        remove = [",", "?"]
    word = word.lower()

    for item in remove:
        word = word.replace(item, "")
    word = word.replace("'s", " 's")

    return word.strip()


class TextDecoder:
    """Base class to be inherited by all decoding strategies. Contains
    implementations that are common for all strategies.

    Args:
        vocab (list): Collection of all words in vocabulary.

    """

    def __init__(self, vocab):
        self._vocab = vocab
        self._vocab_size = vocab.get_size()

        # Lists to store completed sequences and scores
        self._complete_seqs = []
        self._complete_seqs_scores = []

    def init_batch(self, sample_list):
        img_size = sample_list.image_feature_0.size()
        self._batch_size, feature_size_1, feature_size_2 = img_size
        t_batch_size = self._batch_size * self._decode_size
        self.seqs = sample_list.answers.new_full(
            (t_batch_size, 1), self._vocab.SOS_INDEX, dtype=torch.long
        )
        sample_list.image_feature_0 = (
            sample_list.image_feature_0.unsqueeze(1)
            .expand(-1, self._decode_size, -1, -1)
            .reshape(t_batch_size, feature_size_1, feature_size_2)
        )
        self.sample_list = sample_list
        return sample_list

    def add_next_word(self, seqs, prev_word_inds, next_word_inds):
        return torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

    def find_complete_inds(self, next_word_inds):
        incomplete_inds = []
        for ind, next_word in enumerate(next_word_inds):
            if next_word != self._vocab.EOS_INDEX:
                incomplete_inds.append(ind)
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
        return complete_inds, incomplete_inds

    def update_data(self, data, prev_word_inds, next_word_inds, incomplete_inds):
        data["texts"] = next_word_inds[incomplete_inds].unsqueeze(1)
        h1 = data["state"]["td_hidden"][0][prev_word_inds[incomplete_inds]]
        c1 = data["state"]["td_hidden"][1][prev_word_inds[incomplete_inds]]
        h2 = data["state"]["lm_hidden"][0][prev_word_inds[incomplete_inds]]
        c2 = data["state"]["lm_hidden"][1][prev_word_inds[incomplete_inds]]
        data["state"] = {"td_hidden": (h1, c1), "lm_hidden": (h2, c2)}
        return data


def load_str_list(fname):
    with PathManager.open(fname) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    return lines


class VocabDict:
    UNK_TOKEN = "<unk>"
    PAD_TOKEN = "<pad>"
    START_TOKEN = "<s>"
    END_TOKEN = "</s>"

    PAD_INDEX = 0
    SOS_INDEX = 1
    EOS_INDEX = 2
    UNK_INDEX = 3

    def __init__(self, vocab_file, data_dir=None):
        if not os.path.isabs(vocab_file) and data_dir is not None:
            vocab_file = get_absolute_path(os.path.join(data_dir, vocab_file))

        if not PathManager.exists(vocab_file):
            raise RuntimeError(f"Vocab file {vocab_file} for vocab dict doesn't exist")

        self.word_list = load_str_list(vocab_file)
        self._build()

    def _build(self):
        if self.UNK_TOKEN not in self.word_list:
            self.word_list = [self.UNK_TOKEN] + self.word_list

        self.word2idx_dict = {w: n_w for n_w, w in enumerate(self.word_list)}

        # String (word) to integer (index) dict mapping
        self.stoi = self.word2idx_dict
        # Integer to string (word) reverse mapping
        self.itos = self.word_list
        self.num_vocab = len(self.word_list)

        self.UNK_INDEX = (
            self.word2idx_dict[self.UNK_TOKEN]
            if self.UNK_TOKEN in self.word2idx_dict
            else None
        )

        self.PAD_INDEX = (
            self.word2idx_dict[self.PAD_TOKEN]
            if self.PAD_TOKEN in self.word2idx_dict
            else None
        )

    def idx2word(self, n_w):
        return self.word_list[n_w]

    def __len__(self):
        return len(self.word_list)

    def get_size(self):
        return len(self.word_list)

    def get_unk_index(self):
        return self.UNK_INDEX

    def get_unk_token(self):
        return self.UNK_TOKEN

    def word2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.UNK_INDEX is not None:
            return self.UNK_INDEX
        else:
            raise ValueError(
                "word %s not in dictionary \
                             (while dictionary does not contain <unk>)"
                % w
            )

    def tokenize_and_index(self, sentence):
        inds = [self.word2idx(w) for w in tokenize(sentence)]
        return inds
