# Copyright (c) Facebook, Inc. and its affiliates.

import glob
import importlib
import logging
import os
import sys
import warnings

import torch
from src.utils.configuration import get_mmf_env, load_yaml
from src.utils.distributed import is_master, synchronize
from src.utils.download import download_pretrained_model
from src.utils.file_io import PathManager
from omegaconf import OmegaConf


try:
    import git
except ImportError:
    git = None

logger = logging.getLogger(__name__)
ALLOWED_CHECKPOINT_EXTS = [".ckpt", ".pth", ".pt"]


def _hack_imports():
    # NOTE: This can probably be made universal to support backwards
    # compatibility with name "pythia" if needed.
    sys.modules["pythia"] = importlib.import_module("mmf")
    sys.modules["pythia.utils.configuration"] = importlib.import_module(
        "mmf.utils.configuration"
    )


def _load_pretrained_checkpoint(checkpoint_path, *args, **kwargs):
    assert (
        os.path.splitext(checkpoint_path)[1] in ALLOWED_CHECKPOINT_EXTS
    ), f"Checkpoint must have extensions: {ALLOWED_CHECKPOINT_EXTS}"

    _hack_imports()

    with PathManager.open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location=lambda storage, loc: storage)
    assert "config" in ckpt, (
        "No configs provided with pretrained model "
        " while checkpoint also doesn't have configuration."
    )
    config = ckpt.pop("config", None)
    model_config = config.get("model_config", config)

    ckpt = ckpt.get("model", ckpt)

    if "model_name" in kwargs:
        model_name = kwargs["model_name"]
    else:
        assert len(model_config.keys()) == 1, "Only one model type should be specified."
        model_name = list(model_config.keys())[0]

    model_config = model_config.get(model_name)
    return {"config": model_config, "checkpoint": ckpt, "full_config": config}


def _load_pretrained_model(model_name_or_path, *args, **kwargs):
    if PathManager.exists(model_name_or_path):
        download_path = model_name_or_path
        model_name = model_name_or_path
    else:
        download_path = download_pretrained_model(model_name_or_path, *args, **kwargs)
        model_name = model_name_or_path

    configs = glob.glob(os.path.join(download_path, "*.yaml"))
    assert len(configs) <= 1, (
        "Multiple yaml files with the pretrained model. "
        + "MMF doesn't know what to do."
    )

    ckpts = []
    allowed_ckpt_types = [f"*{ext}" for ext in ALLOWED_CHECKPOINT_EXTS]
    for ckpt_type in allowed_ckpt_types:
        ckpts.extend(glob.glob(os.path.join(download_path, ckpt_type)))

    assert (
        len(ckpts) == 1
    ), "None or multiple checkpoints files. MMF doesn't know what to do."

    _hack_imports()

    with PathManager.open(ckpts[0], "rb") as f:
        ckpt = torch.load(f, map_location=lambda storage, loc: storage)
    # If configs are not present, will ckpt provide the config?
    if len(configs) == 0:
        assert "config" in ckpt, (
            "No configs provided with pretrained model"
            " while checkpoint also doesn't have configuration."
        )
        config = ckpt["config"]
    else:
        config = load_yaml(configs[0])
    model_config = config.get("model_config", config)
    ckpt = ckpt.get("model", ckpt)

    # Also handle the case of model_name is path
    if PathManager.exists(model_name):
        # This shouldn't happen
        assert len(model_config.keys()) == 1, "Checkpoint contains more than one model?"
        # Take first key
        model_config = model_config[list(model_config.keys())[0]]
    else:
        model_config = model_config.get(model_name.split(os.path.sep)[-1].split(".")[0])

    return {"config": model_config, "checkpoint": ckpt, "full_config": config}


def load_pretrained_model(model_name_or_path_or_checkpoint, *args, **kwargs):
    # If this is a file, then load this directly else download and load
    if PathManager.isfile(model_name_or_path_or_checkpoint):
        return _load_pretrained_checkpoint(
            model_name_or_path_or_checkpoint, args, kwargs
        )
    else:
        return _load_pretrained_model(model_name_or_path_or_checkpoint, args, kwargs)

