# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from mmf.common.sample import Sample
from mmf.common.registry import registry
from mmf.datasets.mmf_dataset import MMFDataset
from mmf.utils.distributed import byte_tensor_to_object, object_to_byte_tensor
from mmf.utils.pos_emb import pos_emb_calculator
from mmf.utils.text import word_tokenize, mask_tokens
from transformers.tokenization_auto import AutoTokenizer
import copy as c


class TextVQADataset(MMFDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__("textvqa", config, dataset_type, index=imdb_file_index)
        self.use_ocr = self.config.use_ocr
        self.use_ocr_info = self.config.use_ocr_info
        self.use_all_pretrain_data = getattr(self.config, "use_all_pretrain_data", False)
        self.joint_train = registry.get("joint_train")
        self.current_epoch_mode = None
        self.joint_train_mode = registry.get("joint_train_mode", None, no_warning=True)

        self.pos_emb_calculator = pos_emb_calculator(
            Dim=self.config.get("pos_emb_length",20),
            L=self.config.processors.bbox_processor.params.max_length)
        if getattr(self.config, "pretrain", False):
            self.pretrain_mlm = self.config.pretrain.type == 'mlm'
        else:
            self.pretrain_mlm = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True
        )
        print(f'LENGTH of annotation db: {len(self.annotation_db)}')
        print(f'LENGTH of feature db: {len(self.features_db)}')

    
    def preprocess_sample_info(self, sample_info):
        path = self._get_path_based_on_index(self.config, "annotations", self._index)
        # NOTE, TODO: Code duplication w.r.t to STVQA, revisit
        # during dataset refactor to support variable dataset classes
        if "stvqa" in path:
            feature_path = sample_info["feature_path"]
            append = "train"
            if self.dataset_type == "test":
                append = "test_task3"
            if not feature_path.startswith(append):
                feature_path = append + "/" + feature_path
            sample_info["feature_path"] = feature_path
            return sample_info
        # COCO Annotation DBs have corrext feature_path
        elif "COCO" not in sample_info["feature_path"]:
            sample_info["feature_path"] = sample_info["image_path"].replace(
                ".jpg", ".npy"
            )

        # postprocess for joint_train
        if not self.current_epoch_mode:
            return sample_info
        elif self.current_epoch_mode == 'textvqa':
            sample_info = {k: v for k, v in sample_info.items() if "obj_pretrain" not in k}
        elif self.current_epoch_mode == self.joint_train_mode:
            sample_info = {k: v for k, v in sample_info.items() if self.joint_train_mode in k}
            sample_info = {k.replace(f'_{self.joint_train_mode}', ''): v for k, v in sample_info.items()}
        else:
            raise NotImplementedError

        return sample_info


    def postprocess_evalai_entry(self, entry):
        return entry  # Do nothing

    def format_for_prediction(self, report):
        answer_processor = self.answer_processor

        batch_size = len(report.question_id)
        pred_answers = report.scores.argmax(dim=-1).view(batch_size, -1)
        answer_space_size = answer_processor.get_true_vocab_size()

        image_ids = report.image_id.cpu().numpy()
        ocr_source_num = report.ocr_source_num[0]
        context_tokens = []
        for i in range(ocr_source_num):
            context_tokens.append(report[f"context_tokens_{i}"].cpu().numpy())
        ocr_source = report.source.cpu().numpy()
        predictions = []
        for idx, question_id in enumerate(report.question_id):
            # collect VQA answers
            image_id = byte_tensor_to_object(image_ids[idx])
            tokens = byte_tensor_to_object(context_tokens[ocr_source[idx]][idx])
            answer_words = []
            pred_source = []
            for answer_id in pred_answers[idx].tolist():
                if answer_id >= answer_space_size:
                    answer_id -= answer_space_size
                    answer_words.append(word_tokenize(tokens[answer_id]))
                    pred_source.append("OCR")
                else:
                    if answer_id == answer_processor.EOS_IDX:
                        break
                    answer_words.append(
                        answer_processor.answer_vocab.idx2word(answer_id)
                    )
                    pred_source.append("VOCAB")
            # join all the answer tokens with space
            # (this should be correct for almost all cases)
            pred_answer = " ".join(answer_words).replace(" 's", "'s")
            entry = {
                "question_id": question_id.item(),
                "image_id": image_id,
                "answer": pred_answer,
                "pred_source": pred_source,
            }
            entry = self.postprocess_evalai_entry(entry)

            predictions.append(entry)

        return predictions

    def _get_current_epoch_mode(self):
        if self.joint_train:
            return registry.get("current_epoch_mode", no_warning=True)
        else:
            return None

    def _get_id_key(self, orig_key_name, mode):
        # build key name to handle joint_train
        if not mode:
            return orig_key_name
        return f"{orig_key_name}_{mode}"

    def update_current_epoch_mode(self, current_epoch_mode):
        self.current_epoch_mode = current_epoch_mode

    def is_textvqa_train(self):
        if self.current_epoch_mode is None or self.current_epoch_mode == 'textvqa':
            return True
        return False

    def __getitem__(self, idx):
        current_epoch_mode = self._get_current_epoch_mode()
        if current_epoch_mode != self.current_epoch_mode:
            print(f"changing model from {self.current_epoch_mode} to {current_epoch_mode}")
            self.update_current_epoch_mode(current_epoch_mode)
            # build_annotation_db so that the obj pretrain data will include new sampled data
            if self.dataset_type == 'train' and self.use_all_pretrain_data and self.current_epoch_mode != "textvqa":
                self.build_annotation_db()

        sample_info = self.annotation_db[idx]
        sample_info = self.preprocess_sample_info(sample_info)
        current_sample = Sample()

        # breaking change from VQA2Dataset: load question_id
        try:
            current_sample.question_id = torch.tensor(
                sample_info['question_id'], dtype=torch.int
            )
        except Exception as e:
            import pdb; pdb.set_trace()
            print(e)


        if isinstance(sample_info['image_id'], int):
            current_sample.image_id = str(sample_info['image_id'])
        else:
            current_sample.image_id = sample_info['image_id']
        if self._use_features is True:
            # image_xx_0: obj frcnn feat; image_xx_1: ocr frcnn feat
            features = self.features_db[idx]
# <<<<<<< HEAD
#             source_to_use = registry.get('current_epoch', 0) % self.annotation_db.load_file_num +1
#             if f"image_feature_{source_to_use}" in features:
#                 features["image_feature_1"] = features[f"image_feature_{source_to_use}"]
#                 features["image_info_1"] = features[f"image_info_{source_to_use}"]
#
#             # if self.joint_train and current_epoch_mode != "textvqa":
#             #     features["image_feature_0"] = features[f"image_feature_{current_epoch_mode}"]
#             #     features["image_info_0"] = features[f"image_info_{current_epoch_mode}"]
# =======
#             # source_to_use = registry.get('current_epoch', 0) % self.annotation_db.load_file_num +1
#             # if f"image_feature_{source_to_use}" in features:
#             #   features["image_feature_1"] = features[f"image_feature_{source_to_use}"]
#             #    features["image_info_1"] = features[f"image_info_{source_to_use}"]
# >>>>>>> zhen
            current_sample.update(features)

        current_sample = self.add_sample_details(sample_info, current_sample)
        current_sample = self.add_answer_info(sample_info, current_sample)

        # only the 'max_features' key is needed
        # pop other keys to minimize data loading overhead
        if hasattr(current_sample, "image_info_0"):
            for k in list(current_sample.image_info_0):
                if k != "max_features":
                    current_sample.image_info_0.pop(k)
        if hasattr(current_sample, "image_info_1"):
            for k in list(current_sample.image_info_1):
                if k != "max_features":
                    current_sample.image_info_1.pop(k)
        # print("in __getitem__", current_sample.keys())
        #for key in current_sample:
         #   if type(current_sample[key]) is torch.Tensor:
         #       print(key, current_sample[key].size())
        #    if type(current_sample[key]) is list:
        #        print(key, len(current_sample[key]))
        return current_sample

    def add_sample_details(self, sample_info, sample):
        sample.image_id = object_to_byte_tensor(sample.image_id)

        if not self.use_ocr:
            raise NotImplementedError

        if self.joint_train and self.current_epoch_mode != "textvqa":
            # 1. Load text (question words)
            question_str = (
                sample_info["question"]
                if "question" in sample_info
                else sample_info["question_str"]
            )
            text_processor_args = {"text": question_str}

            if "question_tokens" in sample_info:
                text_processor_args["tokens"] = sample_info["question_tokens"]

            processed_question = self.text_processor(text_processor_args)

            if "input_ids" in processed_question:
                sample.text = processed_question["input_ids"]
                sample.text_mask = processed_question["input_mask"]
                sample.text_len = torch.tensor(
                    len(processed_question["tokens"]), dtype=torch.long
                )
            else:
                # For GLoVe based processors
                raise NotImplementedError

            # 2. Load object
            # object bounding box information
            if "obj_normalized_boxes" in sample_info and hasattr(self, "copy_processor"):
                sample.obj_bbox_coordinates = self.copy_processor(
                    {"blob": sample_info["obj_normalized_boxes"]}
                )["blob"]
            # 2.2 Load Object Text

            # Object text information
            if "object_tokens" not in sample["image_info_0"]:
                sample['image_info_0']['object_tokens'] = ["Null" for x in range(sample.obj_bbox_coordinates.shape[0])]
            obj_text_processor_args = {"tokens": sample['image_info_0']['object_tokens']}
            object_tokens = self.obj_text_processor(obj_text_processor_args)
            # TODO: tokenize object tokens and convert to indices
            obj_tokens = sample['image_info_0']['object_tokens']
            sample.obj_max_features = torch.tensor(len(obj_tokens))
            sample.obj_bert_context = object_tokens["input_ids"]
            sample.obj_bert_tokens = object_tokens["tokens"]
            sample.obj_bert_input_mask = object_tokens["input_mask"]
            sample.obj_bert_context_len = torch.tensor(len(object_tokens["tokens"]), dtype=torch.long)

            sample.obj_token_map = []
            sample.combined_obj_token_map = []
            temp_obj_bert_subcontext = []
            cnt = 0;
            obj_ptr = 1;
            combined_ptr = len(processed_question["tokens"])
            while (cnt < len(obj_tokens)):
                sample.obj_token_map.append(obj_ptr)
                sample.combined_obj_token_map.append(combined_ptr)
                tgt_token = obj_tokens[cnt]
                processed_token = self.obj_text_processor.tokenize(tgt_token)
                temp_obj_bert_subcontext.append(object_tokens["input_ids"][obj_ptr])
                obj_ptr += len(processed_token)
                combined_ptr += len(processed_token)
                if obj_ptr >= sample.obj_bert_input_mask.shape[0]:
                    break
                cnt += 1

                # 2.3.2 For each OCR source, get FastText or bert embeddings for OCR tokens
                # sample.ocr_tokens: ocr_tokens after initial token processor
                # sample.context_tokens: ocr_tokens to byte tensor
                # sample.context_feature_0: raw text
                # sample.context_info_0: length of the context
                if self.config.processors.context_processor.type == "fasttext":
                    context = self.context_processor({"tokens": ocr_tokens})
                    # this_sample.context = context["text"]
                    this_sample.ocr_tokens = context["tokens"]
                    this_sample.context_tokens = object_to_byte_tensor(context["tokens"])
                    this_sample.context_feature_0 = context["text"]
                    this_sample.context_info_0 = Sample()
                    this_sample.context_info_0.max_features = context["length"]
                    # Here, the only text that goes through BERT is the question, we directly use the tokenized info as bert input

                elif self.config.processors.context_processor.type == "bert_tokenizer":
                    # Additional Sample attributes for bert tokenizer:
                    # sample.bert_context: processed bert tokens
                    # sample.bert_input_mask: processed input mask
                    # # sample.token_map: indice matching map
                    # this_sample.ocr_tokens = ocr_tokens
                    # this_sample.context_tokens = object_to_byte_tensor(ocr_tokens)
                    # this_sample.context_info_0 = Sample()
                    # this_sample.context_info_0.max_features = torch.tensor(len(ocr_tokens))
                    #
                    # context_processor_args = {}
                    # context_processor_args["text"] = " ".join(ocr_tokens)
                    # context_processor_args["tokens"] = ocr_tokens
                    # processed_context = self.context_processor(context_processor_args)
                    # this_sample.bert_context = processed_context["input_ids"]
                    # this_sample.bert_tokens = processed_context["tokens"]
                    # this_sample.bert_input_mask = processed_context["input_mask"]
                    # # this_sample.bert_context_mask = processed_context["input_mask"]
                    # this_sample.bert_context_len = torch.tensor(len(processed_context["tokens"]), dtype=torch.long)

                    l_tmax = self.config.processors.text_processor.params.max_seq_length
                    l_omax = self.config.processors.obj_text_processor.params.max_seq_length
                    # l_cmax = self.config.processors.context_processor.params.max_seq_length
                    l_t = len(processed_question["tokens"])  # sample.text_len
                    l_o = len(object_tokens["tokens"])  # sample.obj_bert_context_len
                    # l_c = len(processed_context["tokens"])  # this_sample.bert_context_len
                    assert l_t <= l_tmax
                    assert l_o <= l_omax
                    # assert l_c <= l_cmax
                    l_pad = (l_tmax + l_omax) - (
                                l_t + l_o) + 1  # We don't include the [CLS] in obj and ocr tokens, so there is total offset of 2
                    sample.bert_combined = torch.cat([
                        sample.text[:l_t],
                        sample.obj_bert_context[1:l_o],
                        torch.zeros(l_pad, dtype=torch.long)])
                    sample.bert_combined_mask = torch.cat([
                        sample.text_mask[:l_t],
                        sample.obj_bert_input_mask[1:l_o],
                        torch.zeros(l_pad, dtype=torch.long)])


        else:   # textvqa
            # 1. Load object box information
            # object bounding box information
            if "obj_normalized_boxes" in sample_info and hasattr(self, "copy_processor"):
                sample.obj_bbox_coordinates = self.copy_processor(
                    {"blob": sample_info["obj_normalized_boxes"]}
                )["blob"]

            # 2. Load text (question words and ocr tokens)
            # 2.1 Load Question
            # sample.text: processed question tokens, padded
            # sample.text_len: length of processed question
            question_str = (sample_info["question"] if "question" in sample_info else sample_info["question_str"])

            text_processor_args = {"text": question_str}
            if "question_tokens" in sample_info:
                text_processor_args["tokens"] = sample_info["question_tokens"]
            processed_question = self.text_processor(text_processor_args)

            if "input_ids" in processed_question:
                sample.text = processed_question["input_ids"]
                sample.text_mask = processed_question["input_mask"]
                sample.text_len = torch.tensor(len(processed_question["tokens"]), dtype=torch.long)
                # print(f'original: {processed_question["input_ids"]}')
                # print(f'mlm txt: {sample.mlm_txt}')
                # print(f'mlm label: {sample.mlm_labels}')
                # import pdb; pdb.set_trace()
            else:
                # For GLoVe based processors, not sure if supported
                raise NotImplementedError

            # 2.2 Load Object Text

            # Object text information
            if "object_tokens" not in sample["image_info_0"]:
                sample['image_info_0']['object_tokens'] = ["Null" for x in range(sample.obj_bbox_coordinates.shape[0])]
            obj_text_processor_args = {"tokens": sample['image_info_0']['object_tokens']}
            object_tokens = self.obj_text_processor(obj_text_processor_args)
            # TODO: tokenize object tokens and convert to indices
            obj_tokens = sample['image_info_0']['object_tokens']
            sample.obj_max_features = torch.tensor(len(obj_tokens))
            sample.obj_bert_context = object_tokens["input_ids"]
            sample.obj_bert_tokens = object_tokens["tokens"]
            sample.obj_bert_input_mask = object_tokens["input_mask"]
            sample.obj_bert_context_len = torch.tensor(len(object_tokens["tokens"]), dtype=torch.long)

            sample.obj_token_map = []
            sample.combined_obj_token_map = []
            temp_obj_bert_subcontext = []
            cnt = 0
            obj_ptr = 1
            combined_ptr = len(processed_question["tokens"])
            while (cnt < len(obj_tokens)):
                sample.obj_token_map.append(obj_ptr)
                sample.combined_obj_token_map.append(combined_ptr)
                tgt_token = obj_tokens[cnt]
                processed_token = self.obj_text_processor.tokenize(tgt_token)
                temp_obj_bert_subcontext.append(object_tokens["input_ids"][obj_ptr])
                obj_ptr += len(processed_token)
                combined_ptr += len(processed_token)
                if obj_ptr >= sample.obj_bert_input_mask.shape[0]:
                    break
                cnt += 1

            # 2.3 Load OCR Data (Multisource)
            ### Sample: contains text info (the question)
            ### This_sample: contains ocr info (ocr text)

            sample.ocr_source_num = self.annotation_db.load_file_num
            temp_ocr_bert_subcontext = {}
            for current_source in range(self.annotation_db.load_file_num):
                sample.__setattr__(f"ocr_source_{current_source}", Sample())
                this_sample = sample.__getattr__(f"ocr_source_{current_source}")

                # 2.3.1 For each OCR source, preprocess OCR tokens
                if f"ocr_tokens_{current_source}" not in sample_info:
                    ocr_token_source = sample_info[f"ocr_tokens_0"]
                else:
                    ocr_token_source = sample_info[f"ocr_tokens_{current_source}"]
                if hasattr(self, "ocr_token_processor"):
                    ocr_tokens = [self.ocr_token_processor({"text": token})["text"] for token in ocr_token_source]
                else:
                    ocr_tokens = ocr_token_source

                max_len = self.config.processors.answer_processor.params.max_length
                ocr_tokens = ocr_tokens[:max_len]

                if f"ocr_info_{current_source}" not in sample_info:
                    ocr_info = sample_info[f"ocr_info_0"][:max_len]
                else:
                    ocr_info = sample_info[f"ocr_info_{current_source}"][:max_len]

                # 2.3.2 For each OCR source, get FastText or bert embeddings for OCR tokens
                # sample.ocr_tokens: ocr_tokens after initial token processor
                # sample.context_tokens: ocr_tokens to byte tensor
                # sample.context_feature_0: raw text
                # sample.context_info_0: length of the context
                if self.config.processors.context_processor.type == "fasttext":
                    context = self.context_processor({"tokens": ocr_tokens})
                    # this_sample.context = context["text"]
                    this_sample.ocr_tokens = context["tokens"]
                    this_sample.context_tokens = object_to_byte_tensor(context["tokens"])
                    this_sample.context_feature_0 = context["text"]
                    this_sample.context_info_0 = Sample()
                    this_sample.context_info_0.max_features = context["length"]
                    # Here, the only text that goes through BERT is the question, we directly use the tokenized info as bert input

                elif self.config.processors.context_processor.type == "bert_tokenizer":
                    # Additional Sample attributes for bert tokenizer:
                    # sample.bert_context: processed bert tokens
                    # sample.bert_input_mask: processed input mask
                    # sample.token_map: indice matching map
                    this_sample.ocr_tokens = ocr_tokens
                    this_sample.context_tokens = object_to_byte_tensor(ocr_tokens)
                    this_sample.context_info_0 = Sample()
                    this_sample.context_info_0.max_features = torch.tensor(len(ocr_tokens))

                    context_processor_args = {}
                    context_processor_args["text"] = " ".join(ocr_tokens)
                    context_processor_args["tokens"] = ocr_tokens
                    processed_context = self.context_processor(context_processor_args)
                    this_sample.bert_context = processed_context["input_ids"]
                    this_sample.bert_tokens = processed_context["tokens"]
                    this_sample.bert_input_mask = processed_context["input_mask"]
                    # this_sample.bert_context_mask = processed_context["input_mask"]
                    this_sample.bert_context_len = torch.tensor(len(processed_context["tokens"]), dtype=torch.long)

                    l_tmax = self.config.processors.text_processor.params.max_seq_length
                    l_omax = self.config.processors.obj_text_processor.params.max_seq_length
                    l_cmax = self.config.processors.context_processor.params.max_seq_length
                    l_t = len(processed_question["tokens"])  # sample.text_len
                    l_o = len(object_tokens["tokens"])  # sample.obj_bert_context_len
                    l_c = len(processed_context["tokens"])  # this_sample.bert_context_len
                    assert l_t <= l_tmax
                    assert l_o <= l_omax
                    assert l_c <= l_cmax
                    l_pad = (l_tmax + l_cmax + l_omax) - (
                                l_t + l_o + l_c) + 2  # We don't include the [CLS] in obj and ocr tokens, so there is total offset of 2
                    this_sample.bert_combined = torch.cat([
                        sample.text[:l_t],
                        sample.obj_bert_context[1:l_o],
                        this_sample.bert_context[1:l_c],
                        torch.zeros(l_pad, dtype=torch.long)])
                    this_sample.bert_combined_mask = torch.cat([
                        sample.text_mask[:l_t],
                        sample.obj_bert_input_mask[1:l_o],
                        this_sample.bert_input_mask[1:l_c],
                        torch.zeros(l_pad, dtype=torch.long)])

                    # Generate the subtokens and map for ocr text
                    this_sample.context_token_map = []
                    this_sample.combined_context_token_map = []
                    temp_ocr_bert_subcontext[current_source] = []
                    cnt = 0;
                    context_ptr = 1;
                    combined_ptr = l_t + l_o - 1
                    while (cnt < len(ocr_tokens)):
                        this_sample.context_token_map.append(context_ptr)
                        this_sample.combined_context_token_map.append(combined_ptr)
                        tgt_token = ocr_tokens[cnt]
                        processed_token = self.context_processor.tokenize(tgt_token)
                        temp_ocr_bert_subcontext[current_source].append(processed_context["input_ids"][context_ptr])
                        context_ptr += len(processed_token)
                        combined_ptr += len(processed_token)
                        if context_ptr >= this_sample.bert_input_mask.shape[0]:
                            break
                        cnt += 1

                    while (len(this_sample.context_token_map) < len(ocr_tokens)):
                        this_sample.context_token_map.append(l_cmax - 1)
                    while (len(this_sample.combined_context_token_map) < len(ocr_tokens)):
                        this_sample.combined_context_token_map.append(l_tmax + l_omax + l_cmax - 1)

                else:
                    raise NotImplementedError

                # Get PHOC embeddings for OCR tokens
                if hasattr(self, "phoc_processor"):
                    context_phoc = self.phoc_processor({"tokens": ocr_tokens})
                    this_sample.context_feature_1 = context_phoc["text"]
                    this_sample.context_info_1 = Sample()
                    this_sample.context_info_1.max_features = context_phoc["length"]

                # OCR token hierarchy vectors (ZHEN: changed)
                if self.config.get("use_ocr_word_position", False):
                    if len(ocr_info) == 0:
                        vec_arr = np.zeros((len(this_sample.ocr_tokens), 60), dtype=np.int) - 1  # TODO: Magic Number 60
                    elif ("position" not in ocr_info[0]) and "additional_properties" not in ocr_info[0]:
                        vec_arr = np.zeros((len(this_sample.ocr_tokens), 60), dtype=np.int) - 1  # TODO: Magic Number 60
                    else:
                        # To change: fix keystr
                        tmp_keystr = "position" if "position" in ocr_info[0] else "additional_properties"
                        word_pos_arr = np.array([x[tmp_keystr] for x in ocr_info]).reshape(len(ocr_info), -1)
                        l, n = word_pos_arr.shape
                        vec_arr = np.zeros((len(this_sample.ocr_tokens), n), dtype=np.int) - 1
                        vec_arr[:l, :] = word_pos_arr
                        vec_arr = self.pos_emb_calculator.calc(vec_arr).reshape(l, -1)
                    this_sample.ocr_pos_emb = self.copy_processor(
                        {"blob": vec_arr}
                    )["blob"][:max_len]

                # OCR bounding box information
                if f"ocr_normalized_boxes_{current_source}" not in sample_info:
                    box_key = f"ocr_normalized_boxes_0"
                else:
                    box_key = f"ocr_normalized_boxes_{current_source}"

                if box_key in sample_info and hasattr(self, "copy_processor"):
                    # New imdb format: OCR bounding boxes are already pre-computed
                    if len(sample_info[box_key].shape) == 1:
                        sample_info[box_key] = np.tile(sample_info[box_key][:, np.newaxis], (1, 4))
                    this_sample.ocr_bbox_coordinates = self.copy_processor(
                        {"blob": sample_info[box_key]}
                    )["blob"][:max_len]
                elif self.use_ocr_info and info_key in sample_info:
                    # Old imdb format: OCR bounding boxes are computed on-the-fly
                    # from ocr_info
                    raise NotImplementedError
                    '''
                    this_sample.ocr_bbox_coordinates = self.bbox_processor(
                        {"info": sample_info[info_key]}
                    )["bbox"].coordinates
                    '''

            if self.pretrain_mlm:
                # Question text
                if not "input_ids" in processed_question:
                    raise NotImplementedError
                input_ids = processed_question["input_ids"].clone().unsqueeze(0)
                sample.text_mlm, sample.text_mlm_labels = mask_tokens(input_ids,
                                                                      self.tokenizer,
                                                                      self.config.pretrain.mlm_probability)
                # import pdb; pdb.set_trace()
                sample.text_mlm = sample.text_mlm.squeeze()
                sample.text_mlm_labels = sample.text_mlm_labels.squeeze()

                # Object Text
                temp_obj_bert_subcontext = torch.tensor(temp_obj_bert_subcontext)
                input_ids = temp_obj_bert_subcontext.clone().unsqueeze(0)
                temp_obj_bert_subcontext_mlm, temp_obj_bert_subcontext_mlm_labels = mask_tokens(input_ids,
                                                                                                self.tokenizer,
                                                                                                self.config.pretrain.mlm_probability)
                # map back to normal length
                sample.obj_bert_context_mlm = sample.obj_bert_context.clone()
                sample.obj_bert_context_mlm_labels = torch.empty(sample.obj_bert_context_mlm.shape,
                                                                 dtype=torch.long).fill_(-100)
                for i, t in enumerate(sample.obj_token_map):
                    sample.obj_bert_context_mlm[t] = temp_obj_bert_subcontext_mlm[0][i]
                    sample.obj_bert_context_mlm_labels[t] = temp_obj_bert_subcontext_mlm_labels[0][i]

                for current_source in range(self.annotation_db.load_file_num):
                    this_sample = sample.__getattr__(f"ocr_source_{current_source}")
                    temp_ocr_bert_subcontext_tensor = torch.tensor(temp_ocr_bert_subcontext[current_source],
                                                                   dtype=torch.long)
                    input_ids = temp_ocr_bert_subcontext_tensor.clone().unsqueeze(0)
                    temp_ocr_bert_subcontext_mlm, temp_ocr_bert_subcontext_mlm_labels = mask_tokens(input_ids,
                                                                                                    self.tokenizer,
                                                                                                    self.config.pretrain.mlm_probability)
                    # map back to normal length
                    this_sample.bert_context_mlm = this_sample.bert_context.clone()
                    this_sample.bert_context_mlm_labels = torch.empty(this_sample.bert_context_mlm.shape).fill_(-100)
                    for i, t in enumerate(this_sample.token_map):
                        this_sample.bert_context_mlm[t] = temp_ocr_bert_subcontext_mlm[0][i]
                        this_sample.bert_context_mlm_labels[t] = temp_ocr_bert_subcontext_mlm_labels[0][i]




        return sample

    def add_answer_info(self, sample_info, sample):
        # Load real answers from sample_info
        answers = sample_info.get("answers", [])
        answers_to_add = None

        if self.is_textvqa_train():  # original train
            for i in range(sample.ocr_source_num):
                answer_processor_arg = {"answers": answers}

                answer_processor_arg["tokens"] = sample[f"ocr_source_{i}"].pop("ocr_tokens", [])

                # print(f"in add_answer_info, iter {i}:")
                # print("answers:", answers)

                processed_answers = self.answer_processor(answer_processor_arg)

                assert not self.config.fast_read, (
                    "In TextVQADataset, online OCR sampling is incompatible "
                    "with fast_read, so fast_read is currently not supported."
                )
                if i == 0:
                    answers_to_add = c.deepcopy(processed_answers)
                    answers_to_add["train_prev_inds"].unsqueeze_(0)
                else:
                    for key in answers_to_add:
                        if key == "answers":
                            continue
                        if key == "train_prev_inds":
                            answers_to_add["train_prev_inds"] = torch.cat(
                                [answers_to_add[key], processed_answers[key].unsqueeze(0)],
                                0
                            )
                        elif key == "sampled_idx_seq":
                            if answers_to_add[key] is not None:
                                answers_to_add[key] += (-1,) + processed_answers[key]
                        else:
                            answers_to_add[key] = torch.cat(
                                [answers_to_add[key], processed_answers[key]],
                                0
                            )
                    #if i < 10:
                    #    print(i, answers_to_add.keys(), processed_answers.keys())
                    #answers_to_add["answers_scores"] = torch.cat(
                    #    [answers_to_add["answers_scores"], processed_answers["answers_scores"]],
                    #    0
                    #)
                    #answers_to_add["train_prev_inds"] = torch.cat(
                    #    [answers_to_add["train_prev_inds"], processed_answers["train_prev_inds"].unsqueeze(0)],
                    #    0
                    #)
                    #answers_to_add["train_loss_mask"] = torch.cat(
                    #    [answers_to_add["train_loss_mask"], processed_answers["train_loss_mask"]],
                    #    0
                    #)
                    #answers_to_add["sampled_idx_seq"] += (-1,) + processed_answers["sampled_idx_seq"]
                # print("answers_to_add:")
                # print("answers_scores:", answers_to_add["answers_scores"].shape)
                # print("train_prev_inds", answers_to_add["train_prev_inds"].shape)
                # print("train_loss_mask", answers_to_add["train_loss_mask"].shape)

            sample.update(answers_to_add)
            sample.answers = object_to_byte_tensor(answers)
        else:   # obj pretrain
            answer_processor_arg = {"answers": answers}
            processed_answers = answer_processor_arg

            sample.update(processed_answers)
            sample.answers = object_to_byte_tensor(answers)

        if self.is_textvqa_train():
            if not self.pretrain_mlm:
                if "answers_scores" in sample:
                    sample.targets = sample.pop("answers_scores")
            else:
                sample.targets = 0
        elif self.current_epoch_mode == 'obj_pretrain':
            if isinstance(processed_answers["answers"], int):
                sample.targets = torch.tensor(processed_answers["answers"])
        else:
            raise NotImplementedError

        return sample

    def prepare_batch(self, batch):
        for i in range(batch["ocr_source_num"][0]):
            batch[f"context_tokens_{i}"] = batch[f"ocr_source_{i}"].context_tokens
        return super().prepare_batch(batch)