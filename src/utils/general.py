# Copyright (c) Facebook, Inc. and its affiliates.

import collections
import os
from src.utils.file_io import PathManager

import logging
import torch
from torch import nn

from src.utils.distributed import get_rank
logger = logging.getLogger(__name__)


def get_mmf_root():
    from src.common.registry import registry

    mmf_root = registry.get("mmf_root", no_warning=True)
    if mmf_root is None:
        mmf_root = os.path.dirname(os.path.abspath(__file__))
        mmf_root = os.path.abspath(os.path.join(mmf_root, ".."))
        registry.register("mmf_root", mmf_root)
    return mmf_root


def get_absolute_path(paths):
    # String check should be first as Sequence would pass for string too
    if isinstance(paths, str):
        # If path is absolute return it directly
        if os.path.isabs(paths):
            return paths

        possible_paths = [
            # Direct path
            paths
        ]
        # Now, try relative to user_dir if it exists
        from src.utils.configuration import get_mmf_env

        user_dir = get_mmf_env(key="user_dir")
        if user_dir:
            possible_paths.append(os.path.join(user_dir, paths))

        mmf_root = get_mmf_root()
        # Relative to root folder of mmf install
        possible_paths.append(os.path.join(mmf_root, "..", paths))
        # Relative to mmf root
        possible_paths.append(os.path.join(mmf_root, paths))

        # Test all these paths, if any exists return
        for path in possible_paths:
            if PathManager.exists(path):
                # URIs
                if path.find("://") == -1:
                    return os.path.abspath(path)
                else:
                    return path

        # If nothing works, return original path so that it throws an error
        return paths
    elif isinstance(paths, collections.abc.Iterable):
        return [get_absolute_path(path) for path in paths]
    else:
        raise TypeError("Paths passed to dataset should either be " "string or list")


def get_optimizer_parameters(model, config):
    parameters = model.parameters()

    has_custom = hasattr(model, "get_optimizer_parameters")
    if has_custom:
        parameters = model.get_optimizer_parameters(config)

    is_parallel = isinstance(model, nn.DataParallel) or isinstance(
        model, nn.parallel.DistributedDataParallel
    )

    if is_parallel and hasattr(model.module, "get_optimizer_parameters"):
        parameters = model.module.get_optimizer_parameters(config)

    # If parameters are a generator, convert to a list first
    parameters = list(parameters)

    if len(parameters) == 0:
        raise ValueError("optimizer got an empty parameter list")

    # If parameters are in format of list, instead of grouped params
    # convert them to grouped params form
    if not isinstance(parameters[0], dict):
        parameters = [{"params": parameters}]

    for group in parameters:
        group["params"] = list(group["params"])

    check_unused_parameters(parameters, model, config)

    return parameters


def check_unused_parameters(parameters, model, config):
    optimizer_param_set = {p for group in parameters for p in group["params"]}
    unused_param_names = []
    for n, p in model.named_parameters():
        if p.requires_grad and p not in optimizer_param_set:
            unused_param_names.append(n)
    if len(unused_param_names) > 0:
        logger.info(
            "Model parameters not used by optimizer: {}".format(
                " ".join(unused_param_names)
            )
        )
        if not config.optimizer.allow_unused_parameters:
            raise Exception(
                "Found model parameters not used by optimizer. Please check the "
                "model's get_optimizer_parameters and add all parameters. If this "
                "is intended, set optimizer.allow_unused_parameters to True to "
                "ignore it."
            )


def log_device_names():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        logger.info(f"CUDA Device {get_rank()} is: {device_name}")


def get_current_device():
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    else:
        return torch.device("cpu")
