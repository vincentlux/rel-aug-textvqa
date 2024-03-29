# Copyright (c) Facebook, Inc. and its affiliates.

import collections
import gc
import logging
import math
import os
import warnings
import yaml
import numpy as np
from bisect import bisect

import torch
from mmf.utils.distributed import get_rank, get_world_size
from mmf.utils.file_io import PathManager
from torch import nn


logger = logging.getLogger(__name__)


def lr_lambda_update(i_iter, cfg):
    if cfg.training.use_warmup is True and i_iter <= cfg.training.warmup_iterations:
        alpha = float(i_iter) / float(cfg.training.warmup_iterations)
        return cfg.training.warmup_factor * (1.0 - alpha) + alpha
    else:
        idx = bisect(cfg.training.lr_steps, i_iter)
        return pow(cfg.training.lr_ratio, idx)


def clip_gradients(model, i_iter, writer, config, scale=1.0):
    max_grad_l2_norm = config.training.max_grad_l2_norm
    clip_norm_mode = config.training.clip_norm_mode

    if max_grad_l2_norm is not None:
        if clip_norm_mode == "all":
            norm = nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_l2_norm * scale
            )
            if writer is not None:
                writer.add_scalars({"grad_norm": norm}, i_iter)
        else:
            raise NotImplementedError(
                "Clip norm mode %s not implemented" % clip_norm_mode
            )


def ckpt_name_from_core_args(config):
    seed = config.training.seed

    ckpt_name = f"{config.datasets}_{config.model}"

    if seed is not None:
        ckpt_name += f"_{seed:d}"

    return ckpt_name


def foldername_from_config_override(args):
    cfg_override = None
    if hasattr(args, "config_override"):
        cfg_override = args.config_override
    elif "config_override" in args:
        cfg_override = args["config_override"]

    folder_name = ""
    if cfg_override is not None and len(cfg_override) > 0:
        folder_name = str(cfg_override)
        folder_name = folder_name.replace(":", ".").replace("\n", " ")
        folder_name = folder_name.replace("/", "_")
        folder_name = " ".join(folder_name.split())
        folder_name = folder_name.replace(". ", ".").replace(" ", "_")
        folder_name = "_" + folder_name
    return folder_name


def get_mmf_root():
    from mmf.common.registry import registry

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
        from mmf.utils.configuration import get_mmf_env

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


def dict_to_string(dictionary):
    logs = []
    if dictionary is None:
        return ""
    for key, val in dictionary.items():
        if hasattr(val, "item"):
            val = val.item()
        # if key.count('_') == 2:
        #     key = key[key.find('_') + 1:]
        logs.append(f"{key}: {val:.4f}")

    return ", ".join(logs)


def yaml_to_json(yaml_file):
    return yaml.load(yaml_file)


def flatten_json(json_file):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '.')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '.')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(json_file)
    return out


def convert_to_float(value):
    if isinstance(value, float):
        return value
    try:  # try pytorch
        return value.item()
    except:
        try:  # try numpy
            print(value.dtype)
            return np.asscalar(value)
        except:
            raise ValueError('do not know how to convert this number {} to float'.format(value))


def get_overlap_score(candidate, target):
    """Takes a candidate word and a target word and returns the overlap
    score between the two.

    Parameters
    ----------
    candidate : str
        Candidate word whose overlap has to be detected.
    target : str
        Target word against which the overlap will be detected

    Returns
    -------
    float
        Overlap score betwen candidate and the target.

    """
    if len(candidate) < len(target):
        temp = candidate
        candidate = target
        target = temp
    overlap = 0.0
    while len(target) >= 2:
        if target in candidate:
            overlap = len(target)
            return overlap * 1.0 / len(candidate)
        else:
            target = target[:-1]
    return 0.0


def updir(d, n):
    """Given path d, go up n dirs from d and return that path"""
    ret_val = d
    for _ in range(n):
        ret_val = os.path.dirname(ret_val)
    return ret_val


def print_cuda_usage():
    print("Memory Allocated:", torch.cuda.memory_allocated() / (1024 * 1024))
    print("Max Memory Allocated:", torch.cuda.max_memory_allocated() / (1024 * 1024))
    print("Memory Cached:", torch.cuda.memory_cached() / (1024 * 1024))
    print("Max Memory Cached:", torch.cuda.max_memory_cached() / (1024 * 1024))


def get_current_tensors():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                print(type(obj), obj.size())
        except Exception:
            pass


def get_batch_size():
    from mmf.utils.configuration import get_global_config

    batch_size = get_global_config("training.batch_size")

    world_size = get_world_size()

    if batch_size % world_size != 0:
        raise RuntimeError(
            "Batch size {} must be divisible by number "
            "of GPUs {} used.".format(batch_size, world_size)
        )

    return batch_size // world_size


def print_model_parameters(model, return_only=False):
    total_params = sum(p.numel() for p in model.parameters())
    trained_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if not return_only:
        logger.info(
            f"Total Parameters: {total_params}. Trained Parameters: {trained_params}"
        )
    return total_params, trained_params


def get_sizes_list(dim, chunks):
    split_size = (dim + chunks - 1) // chunks
    sizes_list = [split_size] * chunks
    sizes_list[-1] = sizes_list[-1] - (sum(sizes_list) - dim)  # Adjust last
    assert sum(sizes_list) == dim
    if sizes_list[-1] < 0:
        n_miss = sizes_list[-2] - sizes_list[-1]
        sizes_list[-1] = sizes_list[-2]
        for j in range(n_miss):
            sizes_list[-j - 1] -= 1
        assert sum(sizes_list) == dim
        assert min(sizes_list) > 0
    return sizes_list


def get_max_updates(config_max_updates, config_max_epochs, train_loader, update_freq):
    if config_max_updates is None and config_max_epochs is None:
        raise ValueError("Neither max_updates nor max_epochs is specified.")

    if isinstance(train_loader.current_dataset, torch.utils.data.IterableDataset):
        warnings.warn(
            "max_epochs not supported for Iterable datasets. Falling back "
            + "to max_updates."
        )
        return config_max_updates, config_max_epochs

    if config_max_updates is not None and config_max_epochs is not None:
        warnings.warn(
            "Both max_updates and max_epochs are specified. "
            + f"Favoring max_epochs: {config_max_epochs}"
        )

    if config_max_epochs is not None:
        max_updates = math.ceil(len(train_loader) / update_freq) * config_max_epochs
        max_epochs = config_max_epochs
    else:
        max_updates = config_max_updates
        max_epochs = max_updates / len(train_loader)

    return max_updates, max_epochs


def get_chunks(x, sizes):
    out = []
    begin = 0
    for s in sizes:
        y = x.narrow(1, begin, s)
        out.append(y)
        begin += s
    return out


def filter_grads(parameters):
    return [param for param in parameters if param.requires_grad]


def log_device_names():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        logger.info(f"CUDA Device {get_rank()} is: {device_name}")


def assert_iterator_finished(iter):
    try:
        _ = next(iter)
    except StopIteration:
        pass
    else:
        assert False


def get_current_device():
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    else:
        return torch.device("cpu")
