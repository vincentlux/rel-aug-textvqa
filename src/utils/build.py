# Copyright (c) Facebook, Inc. and its affiliates.

import os
import warnings
from enum import Enum
from typing import Any, Dict, Type, Union

import torch
from src.common import typings as mmf_typings
from src.common.registry import registry
from src.utils.distributed import is_dist_initialized
from src.utils.general import get_optimizer_parameters
from src.utils.configuration import Configuration


def build_config(
    configuration: Type[Configuration], *args, **kwargs
) -> mmf_typings.DictConfig:
    """Builder function for config. Freezes the configuration and registers
    configuration object and config DictConfig object to registry.

    Args:
        configuration (Configuration): Configuration object that will be
            used to create the config.

    Returns:
        (DictConfig): A config which is of type Omegaconf.DictConfig
    """
    configuration.freeze()
    config = configuration.get_config()
    registry.register("config", config)
    registry.register("configuration", configuration)

    return config


def build_trainer(config: mmf_typings.DictConfig) -> Any:
    """Builder function for creating a trainer class. Trainer class name
    is picked from the config.

    Args:
        config (DictConfig): Configuration that will be used to create
            the trainer.

    Returns:
        (BaseTrainer): A trainer instance
    """
    trainer_type = config.training.trainer
    trainer_cls = registry.get_trainer_class(trainer_type)
    trainer_obj = trainer_cls(config)

    return trainer_obj


def build_optimizer(model, config):
    optimizer_config = config.optimizer
    if not hasattr(optimizer_config, "type"):
        raise ValueError(
            "Optimizer attributes must have a 'type' key "
            "specifying the type of optimizer. "
            "(Custom or PyTorch)"
        )
    optimizer_type = optimizer_config.type

    if not hasattr(optimizer_config, "params"):
        warnings.warn("optimizer attributes has no params defined, defaulting to {}.")

    params = getattr(optimizer_config, "params", {})

    if hasattr(torch.optim, optimizer_type):
        optimizer_class = getattr(torch.optim, optimizer_type)
    else:
        optimizer_class = registry.get_optimizer_class(optimizer_type)
        if optimizer_class is None:
            raise ValueError(
                "No optimizer class of type {} present in "
                "either torch or registered to registry"
            )

    parameters = get_optimizer_parameters(model, config)

    if optimizer_config.get("enable_state_sharding", False):
        # TODO(vedanuj): Remove once OSS is moved to PT upstream
        try:
            from fairscale.optim.oss import OSS
        except ImportError:
            print(
                "Optimizer state sharding requires fairscale. "
                + "Install using pip install fairscale."
            )
            raise

        assert (
            is_dist_initialized()
        ), "Optimizer state sharding can only be used in distributed mode."
        optimizer = OSS(params=parameters, optim=optimizer_class, **params)
    else:
        optimizer = optimizer_class(parameters, **params)
    return optimizer


def build_lightning_optimizers(model, config):
    optimizer = build_optimizer(model, config)

    if config.training.lr_scheduler:
        lr_scheduler = build_scheduler(optimizer, config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }
    else:
        return optimizer


def build_scheduler(optimizer, config):
    scheduler_config = config.get("scheduler", {})

    if not hasattr(scheduler_config, "type"):
        warnings.warn(
            "No type for scheduler specified even though lr_scheduler is True, "
            "setting default to 'Pythia'"
        )
    scheduler_type = getattr(scheduler_config, "type", "pythia")

    if not hasattr(scheduler_config, "params"):
        warnings.warn("scheduler attributes has no params defined, defaulting to {}.")
    params = getattr(scheduler_config, "params", {})
    scheduler_class = registry.get_scheduler_class(scheduler_type)
    scheduler = scheduler_class(optimizer, **params)

    return scheduler

