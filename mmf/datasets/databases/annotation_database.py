# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import json

import random
import numpy as np
import torch
from tqdm import tqdm, trange
from mmf.utils.file_io import PathManager
from mmf.utils.general import get_absolute_path
from mmf.common.registry import registry
random.seed(1)

class AnnotationDatabase(torch.utils.data.Dataset):
    """
    Dataset for Annotations used in MMF

    TODO: Update on docs sprint
    """

    def __init__(self, config, path, *args, **kwargs):
        super().__init__()
        self.metadata = {}
        self.config = config
        self.start_idx = 0
        paths = []
        for s in path.split(","):
            paths.append(get_absolute_path(s))
        self.load_file_num = 0
        self.joint_train = registry.get("joint_train")
        self.load_annotation_db(paths)
        print("in __init__ of annotation_database\n", self.load_file_num)
        registry.register("ocr_source_num", self.load_file_num)
        print(self.data[1].keys())
        self.post_processing()

    def post_processing(self):
        if 'post_processing' not in self.config:
            return
        if self.config.post_processing.type == 'expand_annotation':
            self._expand_annotation()

    def load_annotation_db(self, paths):
        print(paths)
        for i, path in enumerate(paths):
            if path.find("visdial") != -1 or path.find("visual_dialog") != -1:
                self._load_visual_dialog(path)
            elif path.endswith(".npy"):
                if self.load_file_num == 0:
                    self._load_npy(path)
                else:
                    # assume obj pretrain only appear in the last
                    if self.joint_train and i == len(paths) - 1:
                        self._append_obj_pretrain_npy(path)
                    else:
                        self._append_npy(path)
            elif path.endswith(".jsonl"):
                self._load_jsonl(path)
            elif path.endswith(".json"):
                self._load_json(path)
            else:
                raise ValueError("Unknown file format for annotation db")


    def _expand_annotation(self):
        # expand descriptions (each description becomes one example of self.data)
        print('Start expanding annotation')
        new_data = []
        for d in tqdm(self.data):
            d.pop('all_normalized_boxes')
            d.pop('index_needed')
            new_d = copy.deepcopy(d)
            for idx, desc in enumerate(d['descriptions']):
                question = desc['txt']
                question_id = desc['question_id']
                new_desc = copy.copy(new_d)
                new_desc['question'] = question
                new_desc['question_id'] = question_id
                new_desc['answers'] = idx
                new_desc['image_id'] = new_desc['feature_path']
                new_desc.pop('descriptions')
                new_data.append(new_desc)
        print(f'original data size: {len(self.data)} postprocessed data size: {len(new_data)}')
        self.data = new_data



    def _load_jsonl(self, path):
        with PathManager.open(path, "r") as f:
            db = f.readlines()
            for idx, line in enumerate(db):
                db[idx] = json.loads(line.strip("\n"))
            self.data = db
            self.start_idx = 0

    def _load_npy(self, path):
        print(f"Loading annotations from {path}...")
        with PathManager.open(path, "rb") as f:
            self.db = np.load(f, allow_pickle=True)
        self.start_idx = 0

        if type(self.db) == dict:
            self.metadata = self.db.get("metadata", {})
            self.data = self.db.get("data", [])
        else:
            # TODO: Deprecate support for this
            self.metadata = {"version": 1}
            self.data = self.db
            # Handle old imdb support
            if "image_id" not in self.data[0]:
                self.start_idx = 1

        if "ocr_info" in self.data[self.start_idx].keys():
            for i in trange(self.start_idx, len(self.data)):
                self.data[i][f"ocr_info_{self.load_file_num}"] = copy.deepcopy(self.data[i]["ocr_info"])
                self.data[i][f"ocr_tokens_{self.load_file_num}"] = copy.deepcopy(self.data[i]["ocr_tokens"])
                self.data[i][f"ocr_normalized_boxes_{self.load_file_num}"] = copy.deepcopy(self.data[i]["ocr_normalized_boxes"])

                self.data[i].pop("ocr_info")
                self.data[i].pop("ocr_tokens")
                self.data[i].pop("ocr_normalized_boxes")

        if len(self.data) == 0:
            self.data = self.db
        self.load_file_num = 1

    def _append_npy(self, path):
        print(f"Appending annotations from {path}...")
        with PathManager.open(path, "rb") as f:
            new_db = np.load(f, allow_pickle=True)
        new_start_idx = 0
        if type(new_db) == dict:
            new_data = new_db.get("data", [])
        else:
            new_data = new_db
            if "image_id" not in self.data[0]:
                new_start_idx = 1

        print(len(new_data))

        id2idx = {self.data[i]["question"]+str(self.data[i]["image_id"]): i for i in range(self.start_idx, len(self.data))}

        for i in trange(new_start_idx, len(new_data)):
            idx = id2idx[new_data[i]["question"]+str(new_data[i]["image_id"])]
            self.data[idx][f"ocr_info_{self.load_file_num}"] = new_data[i]["ocr_info"]
            self.data[idx][f"ocr_tokens_{self.load_file_num}"] = new_data[i]["ocr_tokens"]
            self.data[idx][f"ocr_normalized_boxes_{self.load_file_num}"] = new_data[i]["ocr_normalized_boxes"]

        self.load_file_num += 1

    def _append_obj_pretrain_npy(self, path):
        # indicate difference by different joint train task
        # suffix = registry.get('config.model_config.m4c.joint_train.task')
        suffix = registry.get("joint_train_mode")
        print(f"Appending annotations from {path}...")
        with PathManager.open(path, "rb") as f:
            new_db = np.load(f, allow_pickle=True)
        new_start_idx = 0
        if type(new_db) == dict:
            new_data = new_db.get("data", [])
        else:
            new_data = new_db
            if "image_id" not in self.data[0]:
                new_start_idx = 1

        print(f'joint train data: {len(new_data)}...original data: {len(self.data)}')

        new_data = new_data.tolist()
        new_data = random.sample(new_data, len(self.data))
        print(f'reduced joint train data size to {len(new_data)}')

        for idx in trange(new_start_idx, len(new_data)):
            self.data[idx][f'obj_normalized_boxes_{suffix}'] = new_data[idx]["obj_normalized_boxes"]
            self.data[idx][f'answers_{suffix}'] = new_data[idx]['answers']
            self.data[idx][f'feature_path_{suffix}'] = new_data[idx]['feature_path']
            self.data[idx][f'image_id_{suffix}'] = new_data[idx]['image_id']
            self.data[idx][f'image_path_{suffix}'] = new_data[idx]['image_path']
            self.data[idx][f'question_{suffix}'] = new_data[idx]['question']
            self.data[idx][f'question_id_{suffix}'] = new_data[idx]['question_id']

            # self.data[idx][f"ocr_info_{self.load_file_num}"] = new_data[i]["ocr_info"]
            # self.data[idx][f"ocr_tokens_{self.load_file_num}"] = new_data[i]["ocr_tokens"]
            # self.data[idx][f"ocr_normalized_boxes_{self.load_file_num}"] = new_data[i]["ocr_normalized_boxes"]

        # TODO: check if needs to +1 (no need to add bc this is used for rotating ocr frcnn feat)
        # self.load_file_num += 1



    def _load_json(self, path):
        with PathManager.open(path, "r") as f:
            data = json.load(f)
        self.metadata = data.get("metadata", {})
        self.data = data.get("data", [])

        if len(self.data) == 0:
            raise RuntimeError("Dataset is empty")

    def _load_visual_dialog(self, path):
        from mmf.datasets.builders.visual_dialog.database import VisualDialogDatabase

        self.data = VisualDialogDatabase(path)
        self.metadata = self.data.metadata
        self.start_idx = 0

    def __len__(self):
        return len(self.data) - self.start_idx

    def __getitem__(self, idx):
        data = self.data[idx + self.start_idx]

        # Hacks for older IMDBs
        if "answers" not in data:
            if "all_answers" in data and "valid_answers" not in data:
                data["answers"] = data["all_answers"]
            if "valid_answers" in data:
                data["answers"] = data["valid_answers"]

        # TODO: Clean up VizWiz IMDB from copy tokens
        if "answers" in data and not isinstance(data["answers"], int) and data["answers"][-1] == "<copy>":
            data["answers"] = data["answers"][:-1]

        return data

    def get_version(self):
        return self.metadata.get("version", None)
