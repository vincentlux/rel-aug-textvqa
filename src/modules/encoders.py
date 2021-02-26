# Copyright (c) Facebook, Inc. and its affiliates.
import os
import pickle
import re
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torchvision
from src.common.registry import registry
# from src.modules.embeddings import ProjectionEmbedding, TextEmbedding
# from src.modules.hf_layers import BertModelJit
# from src.modules.layers import Identity
# from src.utils.build import build_image_encoder, build_text_encoder
# from src.utils.download import download_pretrained_model
# from src.utils.file_io import PathManager
# from src.utils.general import get_absolute_path
from omegaconf import MISSING, OmegaConf
from torch import nn
from transformers.configuration_auto import AutoConfig
from transformers.modeling_auto import AutoModel


try:
    from detectron2.modeling import ShapeSpec, build_resnet_backbone
except ImportError:
    pass


class Encoder(nn.Module):
    @dataclass
    class Config:
        name: str = MISSING

    @classmethod
    def from_params(cls, **kwargs):
        config = OmegaConf.structured(cls.Config(**kwargs))
        return cls(config)
