# Copyright (c) Facebook, Inc. and its affiliates.

import logging

from torch import nn


logger = logging.getLogger(__name__)

ACT2FN = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
}
