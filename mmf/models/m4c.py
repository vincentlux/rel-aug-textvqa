# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import logging
import math
import copy

import torch
import torch.nn.functional as F
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.layers import ClassifierLayer
from mmf.utils.build import build_image_encoder
from omegaconf import OmegaConf
from torch import nn
from transformers.modeling_bert import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertPreTrainedModel,
    BertOnlyMLMHead,
)

logger = logging.getLogger(__name__)


@registry.register_model("m4c")
class M4C(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.mmt_config = BertConfig(**self.config.mmt)
        self._datasets = registry.get("config").datasets.split(",")

    @classmethod
    def config_path(cls):
        return "configs/models/m4c/defaults.yaml"

    def build(self):
        # modules requiring custom learning rates (usually for finetuning)
        self.finetune_modules = []

        # split model building into several components
        self._build_txt_encoding()
        self._build_obj_encoding()
        self._build_ocr_encoding()
        #self._build_ocrtxt_encoding()
        self._build_mmt()
        self._build_output()

    def _build_encoder_config(self):
        return OmegaConf.create(
            {
                "type": "finetune_faster_rcnn_fpn_fc7",
                "params": {
                    "in_dim": 2048,
                    "weights_file": "models/detectron.defaults/fc7_w.pkl",
                    "bias_file": "models/detectron.defaults/fc7_b.pkl",
                    "model_data_dir": self.config.model_data_dir,
                },
            }
        )

    def _build_txt_encoding(self):
        TEXT_BERT_HIDDEN_SIZE = 768

        self.text_bert_config = BertConfig(**self.config.text_bert)
        if self.config.text_bert_init_from_bert_base:
            self.text_bert = TextBert.from_pretrained(
                "bert-base-uncased", config=self.text_bert_config
            )
            # Use a smaller learning rate on text bert when initializing
            # from BERT_BASE
            self.finetune_modules.append(
                {"module": self.text_bert, "lr_scale": self.config.lr_scale_text_bert}
            )
        else:
            logger.info("NOT initializing text_bert from BERT_BASE")
            self.text_bert = TextBert(self.text_bert_config)

        # if the text bert output dimension doesn't match the
        # multimodal transformer (mmt) hidden dimension,
        # add a linear projection layer between the two
        if self.mmt_config.hidden_size != TEXT_BERT_HIDDEN_SIZE:
            logger.info(
                f"Projecting text_bert output to {self.mmt_config.hidden_size} dim"
            )

            self.text_bert_out_linear = nn.Linear(
                TEXT_BERT_HIDDEN_SIZE, self.mmt_config.hidden_size
            )
        else:
            self.text_bert_out_linear = nn.Identity()
    
    def _build_ocrtxt_encoding(self):
        TEXT_BERT_HIDDEN_SIZE = 768

        self.ocrtext_bert_config = BertConfig(**self.config.text_bert)
        if self.config.text_bert_init_from_bert_base:
            self.ocrtext_bert = TextBert.from_pretrained(
                "bert-base-uncased", config=self.ocrtext_bert_config
            )
            # Use a smaller learning rate on text bert when initializing
            # from BERT_BASE
            self.finetune_modules.append(
                {"module": self.ocrtext_bert, "lr_scale": self.config.lr_scale_text_bert}
            )
        else:
            logger.info("NOT initializing text_bert from BERT_BASE")
            self.ocrtext_bert = TextBert(self.ocrtext_bert_config)

        # if the text bert output dimension doesn't match the
        # multimodal transformer (mmt) hidden dimension,
        # add a linear projection layer between the two
        if self.mmt_config.hidden_size != TEXT_BERT_HIDDEN_SIZE:
            logger.info(
                f"Projecting text_bert output to {self.mmt_config.hidden_size} dim"
            )

            self.ocrtext_bert_out_linear = nn.Linear(
                TEXT_BERT_HIDDEN_SIZE, self.mmt_config.hidden_size
            )
        else:
            self.ocrtext_bert_out_linear = nn.Identity()

    def _build_obj_encoding(self):
        # object appearance feature: Faster R-CNN
        self.obj_faster_rcnn_fc7 = build_image_encoder(
            self._build_encoder_config(), direct_features=True
        )
        # apply smaller lr to pretrained Faster R-CNN fc7
        self.finetune_modules.append(
            {"module": self.obj_faster_rcnn_fc7, "lr_scale": self.config.lr_scale_frcn}
        )
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            self.config.obj.mmt_in_dim, self.mmt_config.hidden_size
        )

        # object location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(4, self.mmt_config.hidden_size)

        self.obj_feat_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.obj_bbox_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.obj_drop = nn.Dropout(self.config.obj.dropout_prob)

    def _build_ocr_encoding(self):
        print(self.config.ocr)
        # hacky way to check if it is mlm pretrain
        self.pretrain_mlm = self.config.losses[0]["type"] == "mlm"
        self.ocr_text_embedding = getattr(self.config.ocr, "text_embedding", "fasttext")
        self.remove_ocr_phoc = getattr(self.config.ocr, "remove_ocr_phoc", False)
        self.remove_ocr_frcn = getattr(self.config.ocr, "remove_ocr_frcn", False)
        self.remove_ocr_posemb = getattr(self.config.ocr, "remove_ocr_posemb", True)
        self.remove_ocr_semantics = getattr(self.config.ocr, "remove_ocr_semantics", False)
        self.remove_ocr_bbox = getattr(self.config.ocr, "remove_ocr_bbox", False)

        # OCR appearance feature: Faster R-CNN
        self.ocr_faster_rcnn_fc7 = build_image_encoder(
            self._build_encoder_config(), direct_features=True
        )
        self.finetune_modules.append(
            {"module": self.ocr_faster_rcnn_fc7, "lr_scale": self.config.lr_scale_frcn}
        )

        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            self.config.ocr.mmt_in_dim, self.mmt_config.hidden_size
        )

        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(4, self.mmt_config.hidden_size)

        self.ocr_feat_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.ocr_bbox_layer_norm = nn.LayerNorm(self.mmt_config.hidden_size)
        self.ocr_drop = nn.Dropout(self.config.ocr.dropout_prob)

    def _build_mmt(self):
        self.mmt = MMT(self.mmt_config)

        # allow specifying a different/scaled lr for multimodal transformer
        self.finetune_modules.append(
            {"module": self.mmt, "lr_scale": self.config.lr_scale_mmt}
        )

    def _build_output(self):
        # dynamic OCR-copying scores with pointer network
        self.ocr_ptr_net = OcrPtrNet(**self.config.classifier.ocr_ptr_net)

        # fixed answer vocabulary scores
        num_choices = registry.get(self._datasets[0] + "_num_final_outputs")
        # remove the OCR copying dimensions in LoRRA's classifier output
        # (OCR copying will be handled separately)
        num_choices -= self.config.classifier.ocr_max_num
        self.classifier = ClassifierLayer(
            self.config.classifier.type,
            in_dim=self.mmt_config.hidden_size,
            out_dim=num_choices,
            **self.config.classifier.params,
        )

        self.answer_processor = registry.get(self._datasets[0] + "_answer_processor")

    def forward(self, sample_list):
        # fwd_results holds intermediate forward pass results
        # TODO possibly replace it with another sample list
        # print("in forward, sample_list.targets\n", sample_list.targets)
        fwd_results = {}
        max_conf = -10000000
        pred_source = None
        scores = None
        updated_target = None
        mlm_labels = None
        if not self.pretrain_mlm:
            target_size = sample_list.targets.size()
            target = sample_list.targets.view(target_size[0], sample_list.ocr_source_num[0], -1, target_size[-1])
            updated_loss_mask = None
            loss_mask = sample_list.train_loss_mask.view(target_size[0], sample_list.ocr_source_num[0], -1)
        for i in range(sample_list["ocr_source_num"][0]):
            sample_list["current_source"] = i
            self._forward_txt_encoding(sample_list, fwd_results)
            self._forward_obj_encoding(sample_list, fwd_results)
            self._forward_ocr_encoding(sample_list, fwd_results)
            self._forward_mmt_and_output(sample_list, fwd_results)
            if self.training:
                if i == 0:
                    scores = fwd_results["scores"]
                    if self.pretrain_mlm:
                        mlm_labels = fwd_results["mlm_labels"]
                else:
                    scores = torch.cat([scores, fwd_results["scores"]], 1)
                    if self.pretrain_mlm:
                        mlm_labels = torch.cat([mlm_labels, fwd_results["mlm_labels"]], 1)
            else:
                if i == 0:
                    if self.pretrain_mlm:
                        scores = fwd_results["scores"]
                        mlm_labels = fwd_results["mlm_labels"]
                    else:
                        max_conf = fwd_results["conf"]
                        scores = fwd_results["scores"]
                        pred_source = torch.zeros_like(max_conf).to(torch.long)
                        updated_target = target[:, 0]
                        updated_loss_mask = loss_mask[:, 0]
                else:
                    if self.pretrain_mlm:
                        scores = torch.cat([scores, fwd_results["scores"]], 1)
                        mlm_labels = torch.cat([mlm_labels, fwd_results["mlm_labels"]], 1)
                    else:
                        current_source = torch.ones_like(pred_source) * i
                        pred_source = torch.where(max_conf > fwd_results["conf"], pred_source, current_source)
                        scores = torch.where((max_conf > fwd_results["conf"]).unsqueeze(-1).unsqueeze(-1), scores, fwd_results["scores"])
                        updated_target = torch.where(
                            (max_conf > fwd_results["conf"]).unsqueeze(-1).unsqueeze(-1),
                            updated_target,
                            target[:, i])
                        updated_loss_mask = torch.where(
                            (max_conf > fwd_results["conf"]).unsqueeze(-1),
                            updated_loss_mask,
                            loss_mask[:, i]
                        )
                        max_conf = torch.where(max_conf > fwd_results["conf"], max_conf, fwd_results["conf"])

        # only keep scores in the forward pass results
        #print("in forward:")
        #print("scores:", scores.shape)
        #print("train_loss_mask:", sample_list.train_loss_mask.shape)
        if self.pretrain_mlm:
            results = {"mlm_scores": scores, "mlm_labels": mlm_labels}
            return results

        if not self.training:
        #    print(pred_source.shape)
        #    print(pred_source)
        #    print(updated_target.shape)
            sample_list.targets = updated_target
            sample_list.train_loss_mask = updated_loss_mask
            for i in range(sample_list["ocr_source_num"][0]):
                sample_list[f"context_tokens_{i}"] = sample_list[f"ocr_source_{i}"].context_tokens
            results = {"scores": scores, "source": pred_source}
        else:
            results = {"scores": scores}
        return results

    def _forward_txt_encoding(self, sample_list, fwd_results):
        current_ocr_source_id = sample_list["current_source"]
        current_source = sample_list[f"ocr_source_{current_ocr_source_id}"]
        if self.pretrain_mlm:
            fwd_results["txt_inds"] = sample_list.text_mlm
            fwd_results["obj_token_inds"] = sample_list.obj_bert_context_mlm
            fwd_results["obj_token_inds_labels"] = sample_list.obj_bert_context_mlm_labels
            fwd_results["ocr_token_inds"] = current_source.bert_context_mlm
            fwd_results["ocr_token_inds_labels"] = current_source.bert_context_mlm_labels
        else:
            fwd_results["txt_inds"] = sample_list.text
            fwd_results["obj_token_inds"] = sample_list.obj_bert_context
            fwd_results["ocr_token_inds"] = current_source.bert_context

        if self.config.ocr.text_embedding == "fasttext":
            # Zhen: This probably doesn't support mlm now... 
            # Text embedding
            fwd_results["text_bert_out"] = self.text_bert(txt_inds=sample_list.text, txt_mask=sample_list.text_mask)
            
            # Object embedding
            obj_rawtextemb = self.text_bert(txt_inds=sample_list.obj_bert_context, txt_mask=sample_list.obj_bert_input_mask)
            s = obj_rawtextemb.shape
            m = 100
            obj_context_cat_ls = []
            if self.pretrain_mlm:
                obj_textemb_labels = torch.empty((s[0], m), dtype=torch.long, device=obj_rawtextemb.device).fill_(-100)
            for i in range(s[0]):
                map_ls = sample_list.obj_token_map[i][:m]
                bert_context_rep = F.pad(obj_rawtextemb[i][map_ls],(0,0,0,m-len(map_ls)),"constant",0)
                obj_context_cat_ls.append(bert_context_rep)
                if self.pretrain_mlm:
                    obj_textemb_labels[i, :len(map_ls)] = fwd_results["obj_token_inds_labels"][i][map_ls]
            fwd_results["obj_textemb"] = torch.stack(obj_context_cat_ls,dim=0)          
            if self.pretrain_mlm:
                fwd_results["obj_bert_context_mlm_labels"] = obj_textemb_labels

            # OCR FastText feature (300-dim)
            ocr_textemb = current_source.context_feature_0
            ocr_textemb = F.normalize(ocr_textemb, dim=-1)
            assert ocr_textemb.size(-1) == 300
            fwd_results["ocr_textemb"] = ocr_textemb

        elif self.config.ocr.text_embedding == "bert":
            encode_concat_flag = getattr(self.config.ocr, "encode_concat", False)
            if encode_concat_flag:
                combined_rawtextemb = self.text_bert(txt_inds=current_source.bert_combined, txt_mask=current_source.bert_combined_mask)

                text_cat_ls, objtext_cat_ls, context_cat_ls = [], [], []
                s = combined_rawtextemb.shape #[bs, L(seq), L(rep)]
                l_q = 20 # Length of question, magic number
                m_o = 100 # Max num of object tokens, magic number
                m_c = 50 # Max number of context tokens, magic number


                if self.pretrain_mlm:
                    obj_textemb_labels = torch.empty((s[0], m), dtype=torch.long, device=obj_rawtextemb.device).fill_(-100)                
                    ocr_textemb_labels = torch.empty((s[0], m), dtype=torch.long, device=ocr_rawtextemb.device).fill_(-100)
                
                for i in range(s[0]):
                    text_token_rep = F.pad(combined_rawtextemb[i][:sample_list.text_len[i]], (0,0,0,l_q-sample_list.text_len[i]),"constant",0) #[bs,l_q,L(rep)]
                    obj_map_ls = sample_list.combined_obj_token_map[i][:m_o]
                    obj_token_rep = F.pad(combined_rawtextemb[i][obj_map_ls],(0,0,0,m_o-len(obj_map_ls)),"constant",0)
                    ocr_map_ls = current_source.combined_context_token_map[i][:m_c]
                    ocr_token_rep = F.pad(combined_rawtextemb[i][ocr_map_ls],(0,0,0,m_c-len(ocr_map_ls)),"constant",0)
                    text_cat_ls.append(text_token_rep)
                    objtext_cat_ls.append(obj_token_rep)
                    context_cat_ls.append(ocr_token_rep)
                    if self.pretrain_mlm:
                        map_ls = sample_list.obj_token_map[i][:m_o]
                        obj_textemb_labels[i, :len(map_ls)] = fwd_results["obj_token_inds_labels"][i][map_ls]
                        map_ls = current_source.context_token_map[i][:m_c]
                        ocr_textemb_labels[i, :len(map_ls)] = fwd_results["ocr_token_inds_labels"][i][map_ls]

                fwd_results["text_bert_out"] = torch.stack(text_cat_ls,dim=0)
                fwd_results["obj_textemb"] = torch.stack(objtext_cat_ls,dim=0)
                fwd_results["ocr_textemb"] = torch.stack(context_cat_ls,dim=0)
                if self.pretrain_mlm:
                    fwd_results["obj_bert_context_mlm_labels"] = obj_textemb_labels
                    fwd_results["bert_context_mlm_labels"] = ocr_textemb_labels
            else:
                # Encode Question and OCR tokens separately
                # Question text embedding
                fwd_results["text_bert_out"] = self.text_bert(txt_inds=sample_list.text, txt_mask=sample_list.text_mask) 
                
                # Object embedding
                obj_rawtextemb = self.text_bert(txt_inds=sample_list.obj_bert_context, txt_mask=sample_list.obj_bert_input_mask)
                s = obj_rawtextemb.shape
                m = 100
                obj_context_cat_ls = []
                if self.pretrain_mlm:
                    obj_textemb_labels = torch.empty((s[0], m), dtype=torch.long, device=obj_rawtextemb.device).fill_(-100)
                for i in range(s[0]):
                    map_ls = sample_list.obj_token_map[i][:m]
                    bert_context_rep = F.pad(obj_rawtextemb[i][map_ls],(0,0,0,m-len(map_ls)),"constant",0)
                    obj_context_cat_ls.append(bert_context_rep)
                    if self.pretrain_mlm:
                        obj_textemb_labels[i, :len(map_ls)] = fwd_results["obj_token_inds_labels"][i][map_ls]
                fwd_results["obj_textemb"] = torch.stack(obj_context_cat_ls,dim=0)
                if self.pretrain_mlm:
                    fwd_results["obj_bert_context_mlm_labels"] = obj_textemb_labels

                # OCR embedding
                ocr_rawtextemb = self.text_bert(txt_inds=current_source.bert_context, txt_mask=current_source.bert_input_mask)
                s = ocr_rawtextemb.shape #[bs, L(seq), L(rep)]
                m = 50 # Magic number
                context_cat_ls = []
                if self.pretrain_mlm:
                    ocr_textemb_labels = torch.empty((s[0], m), dtype=torch.long, device=ocr_rawtextemb.device).fill_(-100)
                for i in range(s[0]):
                    map_ls = current_source.context_token_map[i][:m]
                    ocr_token_rep = F.pad(ocr_rawtextemb[i][map_ls],(0,0,0,m-len(map_ls)),"constant",0)
                    context_cat_ls.append(ocr_token_rep)
                    if self.pretrain_mlm:
                        ocr_textemb_labels[i, :len(map_ls)] = fwd_results["ocr_token_inds_labels"][i][map_ls]
                fwd_results["ocr_textemb"] = torch.stack(context_cat_ls,dim=0)
                if self.pretrain_mlm:
                    fwd_results["bert_context_mlm_labels"] = ocr_textemb_labels
        else:
            raise NotImplementedError
        
        if self.config.ocr.normalize_bert:
            fwd_results["obj_textemb"] = F.normalize(fwd_results["obj_textemb"], dim=-1)
            fwd_results["ocr_textemb"] = F.normalize(fwd_results["ocr_textemb"], dim=-1)
            fwd_results["text_bert_out"] = F.normalize(fwd_results["text_bert_out"], dim=-1)

    def _forward_obj_encoding(self, sample_list, fwd_results):
        # object appearance feature: Faster R-CNN fc7
        obj_fc6 = sample_list.image_feature_0
        obj_fc7 = self.obj_faster_rcnn_fc7(obj_fc6)
        obj_fc7 = F.normalize(obj_fc7, dim=-1)

        obj_feat = torch.cat(
            [fwd_results["obj_textemb"], obj_fc7], dim=-1
        )

        # obj_feat = obj_fc7
        obj_bbox = sample_list.obj_bbox_coordinates
        obj_mmt_in = self.obj_feat_layer_norm(
            self.linear_obj_feat_to_mmt_in(obj_feat)
        ) + self.obj_bbox_layer_norm(self.linear_obj_bbox_to_mmt_in(obj_bbox))
        obj_mmt_in = self.obj_drop(obj_mmt_in)
        fwd_results["obj_mmt_in"] = obj_mmt_in

        # binary mask of valid object vs padding
        obj_nums = sample_list.image_info_0.max_features
        fwd_results["obj_mask"] = _get_mask(obj_nums, obj_mmt_in.size(1))

    def _forward_ocr_encoding(self, sample_list, fwd_results):
        current_ocr_source_id = sample_list["current_source"]
        current_source = sample_list[f"ocr_source_{current_ocr_source_id}"]
        
        # OCR PHOC feature (604-dim)
        ocr_phoc = current_source.context_feature_1
        ocr_phoc = F.normalize(ocr_phoc, dim=-1)
        assert ocr_phoc.size(-1) == 604

        # OCR appearance feature: Faster R-CNN fc7
        image_source = sample_list[f"image_feature_{current_ocr_source_id + 1}"]
        ocr_fc6 = image_source[:, : fwd_results["ocr_textemb"].size(1), :]
        ocr_fc7 = self.ocr_faster_rcnn_fc7(ocr_fc6)
        ocr_fc7 = F.normalize(ocr_fc7, dim=-1)

        if self.remove_ocr_phoc:
            ocr_phoc = torch.zeros_like(ocr_phoc)
        if self.remove_ocr_frcn:
            ocr_fc7 = torch.zeros_like(ocr_fc7)
        ocr_feat = torch.cat(
            [fwd_results["ocr_textemb"], ocr_phoc, ocr_fc7], dim=-1
        )
        if not self.remove_ocr_posemb:
            ocr_feat = torch.cat(
                [ocr_feat,current_source.ocr_pos_emb], dim=-1
            )

        ocr_bbox = current_source.ocr_bbox_coordinates
        if self.remove_ocr_semantics:
            ocr_feat = torch.zeros_like(ocr_feat)
        if self.remove_ocr_bbox:
            ocr_bbox = torch.zeros_like(ocr_bbox)
        ocr_mmt_in = self.ocr_feat_layer_norm(
            self.linear_ocr_feat_to_mmt_in(ocr_feat)
        ) + self.ocr_bbox_layer_norm(self.linear_ocr_bbox_to_mmt_in(ocr_bbox))
        ocr_mmt_in = self.ocr_drop(ocr_mmt_in)
        fwd_results["ocr_mmt_in"] = ocr_mmt_in

        # binary mask of valid OCR vs padding
        ocr_nums = current_source.context_info_0.max_features
        fwd_results["ocr_mask"] = _get_mask(ocr_nums, ocr_mmt_in.size(1))

    def _forward_mmt(self, sample_list, fwd_results):
        # first forward the text BERT layers
        fwd_results["txt_emb"] = self.text_bert_out_linear(fwd_results["text_bert_out"])
        fwd_results["txt_mask"] = sample_list.text_mask
        mmt_results = self.mmt(
            txt_emb=fwd_results["txt_emb"],
            txt_mask=fwd_results["txt_mask"],
            obj_emb=fwd_results["obj_mmt_in"],
            obj_mask=fwd_results["obj_mask"],
            ocr_emb=fwd_results["ocr_mmt_in"],
            ocr_mask=fwd_results["ocr_mask"],
            fixed_ans_emb=self.classifier.module.weight,
            prev_inds=fwd_results["prev_inds"] if not self.pretrain_mlm else None,
            pretrain_mlm=self.pretrain_mlm,
        )
        fwd_results.update(mmt_results)

    def _forward_output(self, sample_list, fwd_results):
        if not self.pretrain_mlm:
            mmt_dec_output = fwd_results["mmt_dec_output"]
            mmt_ocr_output = fwd_results["mmt_ocr_output"]
            ocr_mask = fwd_results["ocr_mask"]

            fixed_scores = self.classifier(mmt_dec_output)
            dynamic_ocr_scores = self.ocr_ptr_net(mmt_dec_output, mmt_ocr_output, ocr_mask)
            scores = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1)
            fwd_results["scores"] = scores
        else:
            mlm_labels = torch.cat([
                sample_list.text_mlm_labels,
                fwd_results["obj_bert_context_mlm_labels"],
                fwd_results["bert_context_mlm_labels"],
            ], dim=1)
            fwd_results["mlm_labels"] = mlm_labels

    def _forward_mmt_and_output(self, sample_list, fwd_results):
        if self.pretrain_mlm:
            self._forward_mmt(sample_list, fwd_results)
            self._forward_output(sample_list, fwd_results)
        else:
            if self.training:
                fwd_results["prev_inds"] = sample_list.train_prev_inds[:, sample_list.current_source,:].clone()
                self._forward_mmt(sample_list, fwd_results)
                self._forward_output(sample_list, fwd_results)
            else:
                dec_step_num = sample_list.train_prev_inds.size(-1)
                # print(self.config.beam_size)
                fwd_results["prev_inds"] = torch.zeros_like(sample_list.train_prev_inds[:, sample_list.current_source,:])
                fwd_results["prev_inds"][:, 0] = self.answer_processor.BOS_IDX
                if self.config.beam_size == 1:
                    # fill prev_inds with BOS_IDX at index 0, and zeros elsewhere

                    # greedy decoding at test time
                    for _ in range(dec_step_num):
                        self._forward_mmt(sample_list, fwd_results)
                        self._forward_output(sample_list, fwd_results)

                        # find the highest scoring output (either a fixed vocab
                        # or an OCR), and add it to prev_inds for auto-regressive
                        # decoding
                        argmax_inds = fwd_results["scores"].argmax(dim=-1)
                        fwd_results["prev_inds"][:, 1:] = argmax_inds[:, :-1]
                    fwd_results["conf"] = torch.sum(torch.max(F.log_softmax(fwd_results["scores"], -1), -1)[0], -1)
                else:
                    batch_size = sample_list.train_prev_inds.size(0)
                    fwd_results["scores"] = None
                    for sample_id in range(batch_size):
                        fwd_results_feed = copy.deepcopy(fwd_results)
                        fwd_results_feed["txt_inds"] = fwd_results_feed["txt_inds"][sample_id: sample_id + 1, :]
                        fwd_results_feed["txt_mask"] = fwd_results_feed["txt_mask"][sample_id: sample_id + 1, :]
                        fwd_results_feed["obj_mmt_in"] = fwd_results_feed["obj_mmt_in"][sample_id: sample_id + 1, :]
                        fwd_results_feed["obj_mask"] = fwd_results_feed["obj_mask"][sample_id: sample_id + 1, :]
                        fwd_results_feed["ocr_mmt_in"] = fwd_results_feed["ocr_mmt_in"][sample_id: sample_id + 1, :]
                        fwd_results_feed["ocr_mask"] = fwd_results_feed["ocr_mask"][sample_id: sample_id + 1, :]
                        fwd_results_feed["prev_inds"] = fwd_results_feed["prev_inds"][sample_id: sample_id + 1, :]
                        sample_list_feed = sample_list
                        self._forward_mmt(sample_list_feed, fwd_results_feed)
                        self._forward_output(sample_list_feed, fwd_results_feed)
                        top_k_value, top_k_inds = \
                            F.log_softmax(fwd_results_feed["scores"][:, 0, :], dim=-1) \
                                .topk(self.config.beam_size)
                        dead = 0
                        seq2keep = fwd_results_feed["prev_inds"][0:1, :1]  # beam_size * decoded
                        seqprob = torch.zeros_like(seq2keep[:, 0]).to(torch.float)  # beam_size
                        cand_score = []
                        fwd_results_feed["txt_inds"] = fwd_results_feed["txt_inds"].repeat(self.config.beam_size, 1)
                        fwd_results_feed["txt_mask"] = fwd_results_feed["txt_mask"].repeat(self.config.beam_size, 1)
                        fwd_results_feed["obj_mmt_in"] = fwd_results_feed["obj_mmt_in"].repeat(self.config.beam_size, 1, 1)
                        fwd_results_feed["obj_mask"] = fwd_results_feed["obj_mask"].repeat(self.config.beam_size, 1)
                        fwd_results_feed["ocr_mmt_in"] = fwd_results_feed["ocr_mmt_in"].repeat(self.config.beam_size, 1, 1)
                        fwd_results_feed["ocr_mask"] = fwd_results_feed["ocr_mask"].repeat(self.config.beam_size, 1)
                        fwd_results_feed["prev_inds"] = fwd_results_feed["prev_inds"].repeat(self.config.beam_size, 1)
                        cand = torch.ones_like(fwd_results_feed["prev_inds"]) * 2
                        allow_size = self.config.beam_size
                        for dec_step in range(dec_step_num):
                            seqprob = seqprob.unsqueeze(-1).repeat(1, allow_size).view(-1)
                            seqprob += top_k_value.view(-1)  # (beam_size * beam_size)
                            dec_len = seq2keep.size(1)
                            seq2keep = seq2keep.repeat(1, allow_size).view(-1, dec_len)
                            # (beam_size * beam_size) * dec_len
                            seq2keep = torch.cat([seq2keep, top_k_inds.view(-1, 1)], -1)
                            beam_k_score, beam_k_inds = seqprob.topk(allow_size)
                            seq2keep = seq2keep[beam_k_inds]
                            seqprob = seqprob[beam_k_inds]
                            nextbatch = torch.zeros_like(fwd_results_feed["prev_inds"])
                            filled = 0
                            new_seqprob = seqprob[seqprob.size(0):]
                            new_seq2keep = seq2keep[seq2keep.size(0):]
                            for i, seq in enumerate(seq2keep):
                                if seq[-1] == self.answer_processor.EOS_IDX or dec_step == dec_step_num - 1:
                                    nextbatch = nextbatch[:nextbatch.size(0) - 1]
                                    cand[dead, :seq.size(0)-1] = seq[1:]
                                    cand_score.append(seqprob[i].item() / (seq.size(0) ** self.config.beam_length_penalty))
                                    dead += 1
                                else:
                                    nextbatch[filled, :seq.size(0)] = seq
                                    new_seqprob = torch.cat([new_seqprob, seqprob[i:i+1]], 0)
                                    new_seq2keep = torch.cat([new_seq2keep, seq2keep[i:i+1]], 0)
                                    filled += 1
                            assert dead + nextbatch.size(0) == self.config.beam_size
                            if dead == self.config.beam_size:
                                break
                            fwd_results_feed["txt_inds"] = fwd_results_feed["txt_inds"][:nextbatch.size(0)]
                            fwd_results_feed["txt_mask"] = fwd_results_feed["txt_mask"][:nextbatch.size(0)]
                            fwd_results_feed["obj_mmt_in"] = fwd_results_feed["obj_mmt_in"][:nextbatch.size(0)]
                            fwd_results_feed["obj_mask"] = fwd_results_feed["obj_mask"][:nextbatch.size(0)]
                            fwd_results_feed["ocr_mmt_in"] = fwd_results_feed["ocr_mmt_in"][:nextbatch.size(0)]
                            fwd_results_feed["ocr_mask"] = fwd_results_feed["ocr_mask"][:nextbatch.size(0)]
                            fwd_results_feed["prev_inds"] = nextbatch
                            seq2keep = new_seq2keep
                            seqprob = new_seqprob
                            allow_size = self.config.beam_size - dead
                            self._forward_mmt(sample_list_feed, fwd_results_feed)
                            self._forward_output(sample_list_feed, fwd_results_feed)
                            top_k_value, top_k_inds = \
                                F.log_softmax(fwd_results_feed["scores"][:, dec_step + 1, :], dim=-1) \
                                    .topk(allow_size)
                        idx = torch.argmax(torch.tensor(cand_score))
                        fwd_results["prev_inds"][sample_id, :] = cand[idx][:]
                        scores = torch.zeros_like(fwd_results_feed["scores"][0:1])
                        for i, word_id in enumerate(cand[idx].view(-1)):
                            scores[0, i, word_id] = 1.
                        if fwd_results["scores"] is None:
                            fwd_results["scores"] = scores
                        else:
                            fwd_results["scores"] = torch.cat([fwd_results["scores"], scores], 0)


    def get_optimizer_parameters(self, config):
        optimizer_param_groups = []

        base_lr = config.optimizer.params.lr
        # collect all the parameters that need different/scaled lr
        finetune_params_set = set()
        for m in self.finetune_modules:
            optimizer_param_groups.append(
                {
                    "params": list(m["module"].parameters()),
                    "lr": base_lr * m["lr_scale"],
                }
            )
            finetune_params_set.update(list(m["module"].parameters()))
        # remaining_params are those parameters w/ default lr
        remaining_params = [
            p for p in self.parameters() if p not in finetune_params_set
        ]
        # put the default lr parameters at the beginning
        # so that the printed lr (of group 0) matches the default lr
        optimizer_param_groups.insert(0, {"params": remaining_params})

        return optimizer_param_groups

    @classmethod
    def update_registry_for_pretrained(cls, config, checkpoint, full_output):
        from omegaconf import OmegaConf

        # Hack datasets using OmegaConf
        datasets = full_output["full_config"].datasets
        dataset = datasets.split(",")[0]
        config_mock = OmegaConf.create({"datasets": datasets})
        registry.register("config", config_mock)
        registry.register(
            f"{dataset}_num_final_outputs",
            # Need to add as it is subtracted
            checkpoint["classifier.module.weight"].size(0)
            + config.classifier.ocr_max_num,
        )
        # Fix this later, when processor pipeline is available
        answer_processor = OmegaConf.create({"BOS_IDX": 1})
        registry.register(f"{dataset}_answer_processor", answer_processor)


class TextBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        encoder_inputs = self.embeddings(txt_inds)
        attention_mask = txt_mask

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs, extended_attention_mask, head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output


class MMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.prev_pred_embeddings = PrevPredEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

    def forward(
            self,
            txt_emb,
            txt_mask,
            obj_emb,
            obj_mask,
            ocr_emb,
            ocr_mask,
            fixed_ans_emb,
            prev_inds,
            pretrain_mlm,
    ):
        if not pretrain_mlm:
            # build embeddings for predictions in previous decoding steps
            # fixed_ans_emb is an embedding lookup table for each fixed vocabulary
            dec_emb = self.prev_pred_embeddings(fixed_ans_emb, ocr_emb, prev_inds)

            # a zero mask for decoding steps, so the encoding steps elements can't
            # attend to decoding steps.
            # A triangular causal mask will be filled for the decoding steps
            # later in extended_attention_mask
            dec_mask = torch.zeros(
                dec_emb.size(0), dec_emb.size(1), dtype=torch.float32, device=dec_emb.device
            )
            # TODO: simply generate masked tokens for these
            encoder_inputs = torch.cat([txt_emb, obj_emb, ocr_emb, dec_emb], dim=1)
            attention_mask = torch.cat([txt_mask, obj_mask, ocr_mask, dec_mask], dim=1)
        else:
            encoder_inputs = torch.cat([txt_emb, obj_emb, ocr_emb], dim=1)
            attention_mask = torch.cat([txt_mask, obj_mask, ocr_mask], dim=1)

        # offsets of each modality in the joint embedding space
        txt_max_num = txt_mask.size(-1)
        obj_max_num = obj_mask.size(-1)
        ocr_max_num = ocr_mask.size(-1)
        if not pretrain_mlm:
            dec_max_num = dec_mask.size(-1)
        txt_begin = 0
        txt_end = txt_begin + txt_max_num
        ocr_begin = txt_max_num + obj_max_num
        ocr_end = ocr_begin + ocr_max_num

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, from_seq_length, to_seq_length]
        # So we can broadcast to
        # [batch_size, num_heads, from_seq_length, to_seq_length]
        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )

        if not pretrain_mlm:
            # decoding step elements can attend to themselves in a causal manner
            extended_attention_mask[:, :, -dec_max_num:, -dec_max_num:] = _get_causal_mask(
                dec_max_num, encoder_inputs.device
            )

        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs, extended_attention_mask, head_mask=head_mask
        )

        mmt_seq_output = encoder_outputs[0]

        if not pretrain_mlm:
            mmt_txt_output = mmt_seq_output[:, txt_begin:txt_end]
            mmt_ocr_output = mmt_seq_output[:, ocr_begin:ocr_end]
            mmt_dec_output = mmt_seq_output[:, -dec_max_num:]
            results = {
                "mmt_seq_output": mmt_seq_output,
                "mmt_txt_output": mmt_txt_output,
                "mmt_ocr_output": mmt_ocr_output,
                "mmt_dec_output": mmt_dec_output,
            }
        else:
            mlm_prediction_scores = self.cls(mmt_seq_output)
            results = {
                "mmt_seq_output": mmt_seq_output,
                "scores": mlm_prediction_scores
            }

        return results


class OcrPtrNet(nn.Module):
    def __init__(self, hidden_size, query_key_size=None):
        super().__init__()

        if query_key_size is None:
            query_key_size = hidden_size
        self.hidden_size = hidden_size
        self.query_key_size = query_key_size

        self.query = nn.Linear(hidden_size, query_key_size)
        self.key = nn.Linear(hidden_size, query_key_size)

    def forward(self, query_inputs, key_inputs, attention_mask):
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        assert extended_attention_mask.dim() == 2
        extended_attention_mask = extended_attention_mask.unsqueeze(1)

        query_layer = self.query(query_inputs)
        if query_layer.dim() == 2:
            query_layer = query_layer.unsqueeze(1)
            squeeze_result = True
        else:
            squeeze_result = False
        key_layer = self.key(key_inputs)

        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scores = scores / math.sqrt(self.query_key_size)
        scores = scores + extended_attention_mask
        if squeeze_result:
            scores = scores.squeeze(1)

        return scores


class PrevPredEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        MAX_DEC_LENGTH = 100
        MAX_TYPE_NUM = 5
        hidden_size = config.hidden_size
        ln_eps = config.layer_norm_eps

        self.position_embeddings = nn.Embedding(MAX_DEC_LENGTH, hidden_size)
        self.token_type_embeddings = nn.Embedding(MAX_TYPE_NUM, hidden_size)

        self.ans_layer_norm = nn.LayerNorm(hidden_size, eps=ln_eps)
        self.ocr_layer_norm = nn.LayerNorm(hidden_size, eps=ln_eps)
        self.emb_layer_norm = nn.LayerNorm(hidden_size, eps=ln_eps)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, ans_emb, ocr_emb, prev_inds):
        assert prev_inds.dim() == 2 and prev_inds.dtype == torch.long
        assert ans_emb.dim() == 2

        batch_size = prev_inds.size(0)
        seq_length = prev_inds.size(1)
        ans_num = ans_emb.size(0)

        # apply layer normalization to both answer embedding and OCR embedding
        # before concatenation, so that they have the same scale
        ans_emb = self.ans_layer_norm(ans_emb)
        ocr_emb = self.ocr_layer_norm(ocr_emb)
        assert ans_emb.size(-1) == ocr_emb.size(-1)
        ans_emb = ans_emb.unsqueeze(0).expand(batch_size, -1, -1)
        ans_ocr_emb_cat = torch.cat([ans_emb, ocr_emb], dim=1)
        raw_dec_emb = _batch_gather(ans_ocr_emb_cat, prev_inds)

        # Add position and type embedding for previous predictions
        position_ids = torch.arange(seq_length, dtype=torch.long, device=ocr_emb.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        # Token type ids: 0 -- vocab; 1 -- OCR
        token_type_ids = prev_inds.ge(ans_num).long()
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = position_embeddings + token_type_embeddings
        embeddings = self.emb_layer_norm(embeddings)
        embeddings = self.emb_dropout(embeddings)
        dec_emb = raw_dec_emb + embeddings

        return dec_emb


def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask


@functools.lru_cache(maxsize=32)
def _get_causal_mask(seq_length, device):
    # generate a lower triangular mask
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i + 1):
            mask[i, j] = 1.0
    return mask


def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    x_flat = x.view(batch_size * length, dim)

    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    inds_flat = batch_offsets + inds
    results = F.embedding(inds_flat, x_flat)
    return results
