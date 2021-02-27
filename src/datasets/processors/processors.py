# Copyright (c) Facebook, Inc. and its affiliates.

"""
The processors exist in MMF to make data processing pipelines in various
datasets as similar as possible while allowing code reuse.

The processors also help maintain proper abstractions to keep only what matters
inside the dataset's code. This allows us to keep the dataset ``__getitem__``
logic really clean and no need about maintaining opinions about data type.
Processors can work on both images and text due to their generic structure.

To create a new processor, follow these steps:

1. Inherit the ``BaseProcessor`` class.
2. Implement ``_call`` function which takes in a dict and returns a dict with
   same keys preprocessed as well as any extra keys that need to be returned.
3. Register the processor using ``@registry.register_processor('name')`` to
   registry where 'name' will be used to refer to your processor later.

In processor's config you can specify ``preprocessor`` option to specify
different kind of preprocessors you want in your dataset.

Let's break down processor's config inside a dataset (VQA2.0) a bit to understand
different moving parts.

Config::

    dataset_config:
      vqa2:
        data_dir: ${env.data_dir}
        processors:
          text_processor:
            type: vocab
            params:
              max_length: 14
              vocab:
                type: intersected
                embedding_name: glove.6B.300d
                vocab_file: vqa2/defaults/extras/vocabs/vocabulary_100k.txt
              preprocessor:
                type: simple_sentence
                params: {}

``BaseDataset`` will init the processors and they will available inside your
dataset with same attribute name as the key name, for e.g. `text_processor` will
be available as `self.text_processor` inside your dataset. As is with every module
in MMF, processor also accept a ``DictConfig`` with a `type` and `params`
attributes. `params` defined the custom parameters for each of the processors.
By default, processor initialization process will also init `preprocessor` attribute
which can be a processor config in itself. `preprocessor` can be then be accessed
inside the processor's functions.

Example::

    from mmf.common.registry import registry
    from mmf.datasets.processors import BaseProcessor

    @registry.register_processor('my_processor')
    class MyProcessor(BaseProcessor):
        def __init__(self, config, *args, **kwargs):
            return

        def __call__(self, item, *args, **kwargs):
            text = item['text']
            text = [t.strip() for t in text.split(" ")]
            return {"text": text}
"""

import collections
import copy
import logging
import os
import random
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Union

import numpy as np
import torch
from src.common.registry import registry
from src.common.typings import ProcessorConfigType
# from src.utils.configuration import get_mmf_cache_dir, get_mmf_env
from src.utils.distributed import is_master, synchronize
from src.utils.file_io import PathManager
# from src.utils.text import VocabDict
# from src.utils.vocab import Vocab, WordToVectorDict


logger = logging.getLogger(__name__)


class Processor:
    """Wrapper class used by MMF to initialized processor based on their
    ``type`` as passed in configuration. It retrieves the processor class
    registered in registry corresponding to the ``type`` key and initializes
    with ``params`` passed in configuration. All functions and attributes of
    the processor initialized are directly available via this class.

    Args:
        config (DictConfig): DictConfig containing ``type`` of the processor to
                             be initialized and ``params`` of that processor.

    """

    def __init__(self, config: ProcessorConfigType, *args, **kwargs):
        if not hasattr(config, "type"):
            raise AttributeError(
                "Config must have 'type' attribute to specify type of processor"
            )
        print(config.type)
        processor_class = registry.get_processor_class(config.type)
        print(processor_class)
        params = {}
        if "params" not in config:
            logger.warning(
                "Config doesn't have 'params' attribute to "
                "specify parameters of the processor "
                f"of type {config.type}. Setting to default {{}}"
            )
        else:
            params = config.params

        self.processor = processor_class(params, *args, **kwargs)

        self._dir_representation = dir(self)

    def __call__(self, item, *args, **kwargs):
        return self.processor(item, *args, **kwargs)

    def __getattr__(self, name):
        if "_dir_representation" in self.__dict__ and name in self._dir_representation:
            return getattr(self, name)
        elif "processor" in self.__dict__ and hasattr(self.processor, name):
            return getattr(self.processor, name)
        else:
            raise AttributeError(f"The processor {name} doesn't exist in the registry.")



class BaseProcessor:
    """Every processor in MMF needs to inherit this class for compatibility
    with MMF. End user mainly needs to implement ``__call__`` function.

    Args:
        config (DictConfig): Config for this processor, containing `type` and
                             `params` attributes if available.

    """

    def __init__(self, config: Dict[str, Any], *args, **kwargs):
        return

    def __call__(self, item: Any, *args, **kwargs) -> Any:
        """Main function of the processor. Takes in a dict and returns back
        a dict

        Args:
            item (Dict): Some item that needs to be processed.

        Returns:
            Dict: Processed dict.

        """
        return item
