# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from multiprocessing.pool import ThreadPool

import tqdm
from mmf.datasets.databases.image_database import ImageDatabase
from mmf.datasets.databases.readers.feature_readers import FeatureReader
from mmf.utils.distributed import is_master
from mmf.utils.general import get_absolute_path
from mmf.common.registry import registry


logger = logging.getLogger(__name__)


class FeaturesDatabase(ImageDatabase):
    def __init__(
        self, config, path, annotation_db=None, feature_key=None, *args, **kwargs
    ):
        super().__init__(config, path, annotation_db, *args, **kwargs)
        self.feature_readers = []
        self.feature_dict = {}
        self.feature_key = config.get("feature_key", "feature_path")
        self.feature_key = feature_key if feature_key else self.feature_key
        self._fast_read = config.get("fast_read", False)
        self.joint_train = registry.get("joint_train")

        path = path.split(",")

        for i, image_feature_dir in enumerate(path):
            # NOTE: assume only the last feature can be joint_train feature
            if self.joint_train and i == len(path) - 1:
                joint_train = True
            else:
                joint_train = False
            feature_reader = FeatureReader(
                base_path=get_absolute_path(image_feature_dir),
                depth_first=config.get("depth_first", False),
                max_features=config.get("max_features", 100),
                joint_train=joint_train,
            )
            self.feature_readers.append(feature_reader)

        self.paths = path
        self.annotation_db = annotation_db
        self._should_return_info = config.get("return_features_info", True)

        if self._fast_read:
            path = ", ".join(path)
            logger.info(f"Fast reading features from {path}")
            logger.info("Hold tight, this may take a while...")
            self._threaded_read()

    def _threaded_read(self):
        elements = [idx for idx in range(1, len(self.annotation_db))]
        pool = ThreadPool(processes=4)

        with tqdm.tqdm(total=len(elements), disable=not is_master()) as pbar:
            for i, _ in enumerate(pool.imap_unordered(self._fill_cache, elements)):
                if i % 100 == 0:
                    pbar.update(100)
        pool.close()

    def _fill_cache(self, idx):
        if self.joint_train:
            raise NotImplementedError("not implemented for joint_train")
        feat_file = self.annotation_db[idx]["feature_path"]
        features, info = self._read_features_and_info(feat_file)
        self.feature_dict[feat_file] = (features, info)

    def _read_features_and_info(self, feat_file):
        # first stores original feat, second stores joint_train feat
        features = []
        infos = []
        current_epoch_mode = registry.get("current_epoch_mode")
        for feature_reader in self.feature_readers:
            # if feature_reader.joint_train:
            #     feature, info = feature_reader.read(feat_file[1])
            # else:
            if self.joint_train:
                if current_epoch_mode == 'textvqa':
                    if feature_reader.joint_train:
                        continue
                else:
                    if not feature_reader.joint_train:
                        continue
            feature, info = feature_reader.read(feat_file[0])

            features.append(feature)
            infos.append(info)

        if not self._should_return_info:
            infos = None
        return features, infos

    def _get_image_features_and_info(self, feat_file):
        # assert isinstance(feat_file, str)
        assert isinstance(feat_file, tuple)
        image_feats, infos = self._read_features_and_info(feat_file)

        return image_feats, infos

    def __len__(self):
        self._check_annotation_db_present()
        return len(self.annotation_db)

    def __getitem__(self, idx):
        self._check_annotation_db_present()
        image_info = self.annotation_db[idx]
        return self.get(image_info)

    def get(self, item):
        feature_path = item.get(self.feature_key, None)
        current_epoch_mode = registry.get("current_epoch_mode")
        if feature_path is None:
            feature_path = self._get_feature_path_based_on_image(item)
        # only read related path
        if self.joint_train and current_epoch_mode != "textvqa":
            joint_train_feature_key = f"{self.feature_key}_{current_epoch_mode}"
            joint_train_feature_path = item.get(joint_train_feature_key, None)
            feature_path = tuple((joint_train_feature_path,))
        else:
            feature_path = tuple((feature_path,))

        return self.from_path(feature_path)

    def from_path(self, path):
        # assert isinstance(path, str)
        assert isinstance(path, tuple)

        features, infos = self._get_image_features_and_info(path)

        item = {}
        for idx, image_feature in enumerate(features):
            # if idx == len(features) - 1 and current_epoch_mode != "textvqa":
            #     joint_train_mode = registry.get("joint_train_mode")
            #     item["image_feature_%s" % joint_train_mode] = image_feature
            #     if infos is not None:
            #         item["image_info_%s" % joint_train_mode] = infos[idx]
            # else:
            item["image_feature_%s" % idx] = image_feature
            if infos is not None:
                item["image_info_%s" % idx] = infos[idx]
        return item

    def _get_feature_path_based_on_image(self, item):
        image_path = self._get_attrs(item)[0]
        if isinstance(image_path, int):
            return f"{image_path}.npy"
        feature_path = ".".join(image_path.split(".")[:-1]) + ".npy"
        return feature_path
