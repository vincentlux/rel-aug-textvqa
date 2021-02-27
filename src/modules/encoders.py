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
from src.modules.layers import Identity
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


class EncoderFactory(nn.Module):
    @dataclass
    class Config:
        type: str = MISSING
        params: Encoder.Config = MISSING


class ImageEncoderTypes(Enum):
    default = "default"
    identity = "identity"
    torchvision_resnet = "torchvision_resnet"
    resnet152 = "resnet152"
    detectron2_resnet = "detectron2_resnet"


class ImageEncoderFactory(EncoderFactory):
    @dataclass
    class Config(EncoderFactory.Config):
        type: ImageEncoderTypes = MISSING

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__()
        self._type = config.type

        if isinstance(self._type, ImageEncoderTypes):
            self._type = self._type.value

        params = config.params

        if self._type == "default" or self._type == "identity":
            self.module = nn.Identity()
            self.module.out_dim = params.in_dim
        elif self._type == "resnet152":
            self.module = ResNet152ImageEncoder(params)
        elif self._type == "torchvision_resnet":
            self.module = TorchvisionResNetImageEncoder(params)
        elif self._type == "detectron2_resnet":
            self.module = Detectron2ResnetImageEncoder(params)
        else:
            raise NotImplementedError("Unknown Image Encoder: %s" % self._type)

    @property
    def out_dim(self):
        return self.module.out_dim

    def forward(self, image):
        return self.module(image)


# Taken from facebookresearch/mmbt with some modifications
@registry.register_encoder("resnet152")
class ResNet152ImageEncoder(Encoder):
    @dataclass
    class Config(Encoder.Config):
        name: str = "resnet152"
        pretrained: bool = True
        # "avg" or "adaptive"
        pool_type: str = "avg"
        num_output_features: int = 1

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__()
        self.config = config
        model = torchvision.models.resnet152(pretrained=config.get("pretrained", True))
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        pool_func = (
            nn.AdaptiveAvgPool2d if config.pool_type == "avg" else nn.AdaptiveMaxPool2d
        )

        # -1 will keep the original feature size
        if config.num_output_features == -1:
            self.pool = nn.Identity()
        elif config.num_output_features in [1, 2, 3, 5, 7]:
            self.pool = pool_func((config.num_output_features, 1))
        elif config.num_output_features == 4:
            self.pool = pool_func((2, 2))
        elif config.num_output_features == 6:
            self.pool = pool_func((3, 2))
        elif config.num_output_features == 8:
            self.pool = pool_func((4, 2))
        elif config.num_output_features == 9:
            self.pool = pool_func((3, 3))

        self.out_dim = 2048

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out  # BxNx2048

@registry.register_encoder("torchvision_resnet")
class TorchvisionResNetImageEncoder(Encoder):
    @dataclass
    class Config(Encoder.Config):
        name: str = "resnet50"
        pretrained: bool = False
        zero_init_residual: bool = True

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__()
        self.config = config

        model = getattr(torchvision.models, config.name)(
            pretrained=config.pretrained, zero_init_residual=config.zero_init_residual
        )
        # Set avgpool and fc layers in torchvision to Identity.
        model.avgpool = Identity()
        model.fc = Identity()

        self.model = model
        self.out_dim = 2048

    def forward(self, x):
        # B x 3 x 224 x 224 -> B x 2048 x 7 x 7
        out = self.model(x)
        return out


@registry.register_encoder("detectron2_resnet")
class Detectron2ResnetImageEncoder(Encoder):
    @dataclass
    class Config(Encoder.Config):
        name: str = "detectron2_resnet"
        pretrained: bool = True
        pretrained_path: str = None

    def __init__(self, config: Config, *args, **kwargs):
        super().__init__()
        self.config = config
        pretrained = config.get("pretrained", False)
        pretrained_path = config.get("pretrained_path", None)

        self.resnet = build_resnet_backbone(config, ShapeSpec(channels=3))

        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(
                pretrained_path, progress=False
            )
            new_state_dict = OrderedDict()
            replace_layer = {"backbone.": ""}

            for key, value in state_dict["model"].items():
                new_key = re.sub(
                    r"(backbone\.)", lambda x: replace_layer[x.groups()[0]], key
                )
                new_state_dict[new_key] = value
            self.resnet.load_state_dict(new_state_dict, strict=False)

        self.out_dim = 2048

    def forward(self, x):
        x = self.resnet(x)
        return x["res5"]
