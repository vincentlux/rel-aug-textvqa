# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
from mmf.common.sample import Sample
from mmf.common.registry import registry
from mmf.datasets.mmf_dataset import MMFDataset
from mmf.utils.distributed import byte_tensor_to_object, object_to_byte_tensor
from mmf.utils.text import word_tokenize, mask_tokens
from transformers.tokenization_auto import AutoTokenizer
import copy as c


class TextVQADataset(MMFDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__("textvqa", config, dataset_type, index=imdb_file_index)
        self.use_ocr = self.config.use_ocr
        self.use_ocr_info = self.config.use_ocr_info
        self.pretrain_mlm = self.config.pretrain.type == 'mlm'
        self.tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True
        )

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
        return sample_info

    def postprocess_evalai_entry(self, entry):
        return entry  # Do nothing

    def format_for_prediction(self, report):
        answer_processor = self.answer_processor

        batch_size = len(report.question_id)
        pred_answers = report.scores.argmax(dim=-1).view(batch_size, -1)
        answer_space_size = answer_processor.get_true_vocab_size()

        image_ids = report.image_id.cpu().numpy()
        context_tokens = report.context_tokens.cpu().numpy()
        predictions = []
        for idx, question_id in enumerate(report.question_id):
            # collect VQA answers
            image_id = byte_tensor_to_object(image_ids[idx])
            tokens = byte_tensor_to_object(context_tokens[idx])
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

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        sample_info = self.preprocess_sample_info(sample_info)
        current_sample = Sample()

        # breaking change from VQA2Dataset: load question_id
        current_sample.question_id = torch.tensor(
            sample_info["question_id"], dtype=torch.int
        )

        if isinstance(sample_info["image_id"], int):
            current_sample.image_id = str(sample_info["image_id"])
        else:
            current_sample.image_id = sample_info["image_id"]
        if self._use_features is True:
            # image_xx_0: obj frcnn feat; image_xx_1: ocr frcnn feat
            features = self.features_db[idx]
            # source_to_use = registry.get('current_epoch', 0) % self.annotation_db.load_file_num +1
            # if f"image_feature_{source_to_use}" in features:
            #   features["image_feature_1"] = features[f"image_feature_{source_to_use}"]
            #    features["image_info_1"] = features[f"image_info_{source_to_use}"]
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
        return current_sample

    def add_sample_details(self, sample_info, sample):
        sample.image_id = object_to_byte_tensor(sample.image_id)

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
            sample.text_len = torch.tensor(
                len(processed_question["tokens"]), dtype=torch.long
            )
            if self.pretrain_mlm:
                input_ids = processed_question["input_ids"].clone().unsqueeze(0)
                sample.text_mlm, sample.text_mlm_labels = mask_tokens(input_ids,
                                                                self.tokenizer, self.config.pretrain.mlm_probability)
                # import pdb; pdb.set_trace()
                sample.text_mlm = sample.text_mlm.squeeze()
                sample.text_mlm_labels = sample.text_mlm_labels.squeeze()
            # print(f'original: {processed_question["input_ids"]}')
            # print(f'mlm txt: {sample.mlm_txt}')
            # print(f'mlm label: {sample.mlm_labels}')
            # import pdb; pdb.set_trace()

        else:
            # For GLoVe based processors
            sample.text = processed_question["text"]
            sample.text_len = processed_question["length"]

        # 2. Load object
        # object bounding box information
        if "obj_normalized_boxes" in sample_info and hasattr(self, "copy_processor"):
            sample.obj_bbox_coordinates = self.copy_processor(
                {"blob": sample_info["obj_normalized_boxes"]}
            )["blob"]
        # object text information
        obj_text_processor_args = {"tokens": sample['image_info_0']['object_tokens']}
        object_tokens = self.obj_text_processor(obj_text_processor_args)
        # TODO: tokenize object tokens and convert to indices
        obj_tokens = sample['image_info_0']['object_tokens']
        sample.obj_max_features = torch.tensor(len(obj_tokens))
        sample.obj_bert_context = object_tokens["input_ids"]
        sample.obj_bert_tokens = object_tokens["tokens"]
        sample.obj_bert_input_mask = object_tokens["input_mask"]
        sample.obj_token_map = []
        temp_obj_bert_subcontext = []
        cnt = 0; ptr = 1
        while (cnt < len(obj_tokens)):
            sample.obj_token_map.append(ptr)
            tgt_token = obj_tokens[cnt]
            processed_token = self.obj_text_processor.tokenize(tgt_token)
            temp_obj_bert_subcontext.append(object_tokens["input_ids"][ptr])
            ptr += len(processed_token)
            if ptr >= sample.obj_bert_input_mask.shape[0]:
                break
            cnt += 1

        # only probability over first subtoken
        if self.pretrain_mlm:
            temp_obj_bert_subcontext = torch.tensor(temp_obj_bert_subcontext)
            input_ids = temp_obj_bert_subcontext.clone().unsqueeze(0)
            temp_obj_bert_subcontext_mlm, temp_obj_bert_subcontext_mlm_labels = mask_tokens(input_ids,
                                                            self.tokenizer, self.config.pretrain.mlm_probability)
            # map back to normal length
            sample.obj_bert_context_mlm = sample.obj_bert_context.clone()
            sample.obj_bert_context_mlm_labels = torch.empty(sample.obj_bert_context_mlm.shape, dtype=torch.long).fill_(-100)
            for i, t in enumerate(sample.obj_token_map):
                sample.obj_bert_context_mlm[t] = temp_obj_bert_subcontext_mlm[0][i]
                sample.obj_bert_context_mlm_labels[t] = temp_obj_bert_subcontext_mlm_labels[0][i]

        # 3. Load OCR
        if not self.use_ocr:
            # remove all OCRs from the sample
            # (i.e. make an empty OCR list)
            sample_info["ocr_tokens"] = []
            sample_info["ocr_info"] = []
            if "ocr_normalized_boxes" in sample_info:
                sample_info["ocr_normalized_boxes"] = np.zeros((0, 4), np.float32)
            # clear OCR visual features
            if "image_feature_1" in sample:
                sample.image_feature_1 = torch.zeros_like(sample.image_feature_1)
            return sample

        # Preprocess OCR tokens
        sample.ocr_source_num = self.annotation_db.load_file_num
        for current_source in range(self.annotation_db.load_file_num):
            sample.__setattr__(f"ocr_source_{current_source}", Sample())
            this_sample = sample.__getattr__(f"ocr_source_{current_source}")
            if f"ocr_tokens_{current_source}" not in sample_info:
                ocr_token_source = sample_info[f"ocr_tokens_0"]
            else:
                ocr_token_source = sample_info[f"ocr_tokens_{current_source}"]

            if hasattr(self, "ocr_token_processor"):
                ocr_tokens = [
                    self.ocr_token_processor({"text": token})["text"]
                    for token in ocr_token_source
                ]
            else:
                ocr_tokens = ocr_token_source
            # Get FastText or bert embeddings for OCR tokens
            # TO CHANGE!!!!!!!
            ocr_tokens = ocr_tokens[:self.config.processors.bbox_processor.params.max_length]
            if self.config.processors.context_processor.type == "fasttext":
                context = self.context_processor({"tokens": ocr_tokens})
                this_sample.context = context["text"]
                this_sample.ocr_tokens = context["tokens"]
                #print(sample.ocr_tokens)
                #raise NotImplementedError
                this_sample.context_tokens = object_to_byte_tensor(context["tokens"])
                this_sample.context_feature_0 = context["text"]
                this_sample.context_info_0 = Sample()
                this_sample.context_info_0.max_features = context["length"]
            elif self.config.processors.context_processor.type == "bert_tokenizer":
                context_processor_args = {}
                context_processor_args["text"] = " ".join(ocr_tokens)
                context_processor_args["tokens"] = ocr_tokens
                processed_context = self.context_processor(context_processor_args)
                this_sample.ocr_tokens = ocr_tokens
                this_sample.context_tokens = object_to_byte_tensor(ocr_tokens)
                this_sample.context_info_0 = Sample()
                this_sample.context_info_0.max_features = torch.tensor(len(ocr_tokens))
                this_sample.bert_context = processed_context["input_ids"]
                this_sample.bert_tokens = processed_context["tokens"]
                this_sample.bert_input_mask = processed_context["input_mask"]
                this_sample.token_map = []
                temp_bert_subcontext = []
                cnt = 0; ptr = 1
                while(cnt<len(ocr_tokens)):
                    this_sample.token_map.append(ptr)
                    tgt_token = ocr_tokens[cnt]
                    processed_token = self.context_processor.tokenize(tgt_token)
                    temp_bert_subcontext.append(processed_context["input_ids"][ptr])
                    ptr += len(processed_token)
                    if ptr>= this_sample.bert_input_mask.shape[0]:
                        break
                    cnt+=1
                # only probability over first subtoken
                if self.pretrain_mlm:
                    temp_bert_subcontext = torch.tensor(temp_bert_subcontext, dtype=torch.long)
                    input_ids = temp_bert_subcontext.clone().unsqueeze(0)
                    temp_bert_subcontext_mlm, temp_bert_subcontext_mlm_labels = mask_tokens(input_ids,
                                                                                  self.tokenizer,
                                                                                  self.config.pretrain.mlm_probability)
                    # map back to normal length
                    this_sample.bert_context_mlm = this_sample.bert_context.clone()
                    this_sample.bert_context_mlm_labels = torch.empty(this_sample.bert_context_mlm.shape).fill_(-100)
                    for i, t in enumerate(this_sample.token_map):
                        this_sample.bert_context_mlm[t] = temp_bert_subcontext_mlm[0][i]
                        this_sample.bert_context_mlm_labels[t] = temp_bert_subcontext_mlm_labels[0][i]
                #while(len(sample.token_map)<len(ocr_tokens)):
                #    sample.token_map.append(-1)
            else:
                raise NotImplementedError

            # Get PHOC embeddings for OCR tokens
            if hasattr(self, "phoc_processor"):
                context_phoc = self.phoc_processor({"tokens": ocr_tokens})
                this_sample.context_feature_1 = context_phoc["text"]
                this_sample.context_info_1 = Sample()
                this_sample.context_info_1.max_features = context_phoc["length"]
            # OCR order vectors (ZHEN: removed)
            '''
            if self.config.get("use_order_vectors", False):
                order_vectors = np.eye(len(sample.ocr_tokens), dtype=np.float32)
                order_vectors = torch.from_numpy(order_vectors)
                order_vectors[context["length"] :] = 0
                sample.order_vectors = order_vectors
            '''
            # OCR bounding box information
            if f"ocr_normalized_boxes_{current_source}" not in sample_info:
                box_key = f"ocr_normalized_boxes_0"
            else:
                box_key = f"ocr_normalized_boxes_{current_source}"
            if f"ocr_info_{current_source}" not in sample_info:
                info_key = f"ocr_info_0"
            else:
                info_key = f"ocr_info_{current_source}"

            if box_key in sample_info and hasattr(self, "copy_processor"):
                # New imdb format: OCR bounding boxes are already pre-computed
                max_len = self.config.processors.answer_processor.params.max_length
                this_sample.ocr_bbox_coordinates = self.copy_processor(
                    {"blob": sample_info[box_key]}
                )["blob"][:max_len]
            elif self.use_ocr_info and info_key in sample_info:
                # Old imdb format: OCR bounding boxes are computed on-the-fly
                # from ocr_info
                this_sample.ocr_bbox_coordinates = self.bbox_processor(
                    {"info": sample_info[info_key]}
                )["bbox"].coordinates
        # print("in add_sample_details:, sample:\n\t", sample.keys())
        # print("in add_sample_details:, sample.ocr_source_0:\n\t", sample.ocr_source_0.keys())
        # print("in add_sample_details:, sample.ocr_source_1:\n\t", sample["ocr_source_1"].keys())
        return sample

    def add_answer_info(self, sample_info, sample):
        # Load real answers from sample_info
        answers = sample_info.get("answers", [])
        answers_to_add = None
        for i in range(sample.ocr_source_num):
            answer_processor_arg = {"answers": answers}

            answer_processor_arg["tokens"] = sample[f"ocr_source_{i}"].pop("ocr_tokens", [])

            #print(f"in add_answer_info, iter {i}:")
            #print("answers:", answers)

            processed_answers = self.answer_processor(answer_processor_arg)

            assert not self.config.fast_read, (
                "In TextVQADataset, online OCR sampling is incompatible "
                "with fast_read, so fast_read is currently not supported."
            )
            if i == 0:
                answers_to_add = c.deepcopy(processed_answers)
                answers_to_add["train_prev_inds"].unsqueeze_(0)
            else:
                answers_to_add["answers_scores"] = torch.cat(
                    [answers_to_add["answers_scores"], processed_answers["answers_scores"]],
                    0
                )
                answers_to_add["train_prev_inds"] = torch.cat(
                    [answers_to_add["train_prev_inds"], processed_answers["train_prev_inds"].unsqueeze(0)],
                    0
                )
                answers_to_add["train_loss_mask"] = torch.cat(
                    [answers_to_add["train_loss_mask"], processed_answers["train_loss_mask"]],
                    0
                )
                answers_to_add["sampled_idx_seq"] += (-1, ) + processed_answers["sampled_idx_seq"]
            #print("answers_to_add:")
            #print("answers_scores:", answers_to_add["answers_scores"].shape)
            #print("train_prev_inds", answers_to_add["train_prev_inds"].shape)
            #print("train_loss_mask", answers_to_add["train_loss_mask"].shape)

        sample.update(answers_to_add)
        sample.answers = object_to_byte_tensor(answers)

        if not self.pretrain_mlm:
            if "answers_scores" in sample:
                sample.targets = sample.pop("answers_scores")
        else:
            sample.targets = 0
        return sample
