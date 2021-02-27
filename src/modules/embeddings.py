# Copyright (c) Facebook, Inc. and its affiliates.
# TODO: Update kwargs with defaults

import os
import pickle
from copy import deepcopy
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import torch
# from src.modules.attention import AttentionLayer, SelfAttention, SelfGuidedAttention
# from src.modules.bottleneck import MovieBottleneck
# from src.modules.layers import AttnPool1d, Identity
# from src.utils.file_io import PathManager
# from src.utils.vocab import Vocab
from torch import Tensor, nn
from transformers.modeling_bert import BertEmbeddings

class ProjectionEmbedding(nn.Module):
    def __init__(self, module, in_dim, out_dim, **kwargs):
        super().__init__()
        if module == "linear":
            self.layers = nn.Linear(in_dim, out_dim)
            self.out_dim = out_dim
        elif module == "conv":
            last_out_channels = in_dim
            layers = []
            for conv in kwargs["convs"]:
                layers.append(nn.Conv1d(in_channels=last_out_channels, **conv))
                last_out_channels = conv["out_channels"]
            self.layers = nn.ModuleList(*layers)
            self.out_dim = last_out_channels
        else:
            raise TypeError(
                "Unknown module type for 'ProjectionEmbedding',"
                "use either 'linear' or 'conv'"
            )

    def forward(self, x):
        return self.layers(x)

