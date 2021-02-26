# Copyright (c) Facebook, Inc. and its affiliates.

import glob
import importlib
import logging
import random
import os
import sys
from datetime import datetime

import numpy as np
import torch
from src.utils.general import get_absolute_path


def set_seed(seed):
    if seed:
        if seed == -1:
            # From detectron2
            seed = (
                os.getpid()
                + int(datetime.now().strftime("%S%f"))
                + int.from_bytes(os.urandom(2), "big")
            )
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    return seed


def import_user_module(user_dir: str):
    """Given a user dir, this function imports it as a module.

    This user_module is expected to have an __init__.py at its root.
    You can use import_files to import your python files easily in
    __init__.py

    Args:
        user_dir (str): directory which has to be imported
    """
    logger = logging.getLogger(__name__)
    if user_dir:
        user_dir = get_absolute_path(user_dir)
        module_parent, module_name = os.path.split(user_dir)

        if module_name in sys.modules:
            module_bak = sys.modules[module_name]
            del sys.modules[module_name]
        else:
            module_bak = None

        logger.info(f"Importing from {user_dir}")
        sys.path.insert(0, module_parent)
        importlib.import_module(module_name)

        sys.modules["mmf_user_dir"] = sys.modules[module_name]
        if module_bak is not None and module_name != "mmf_user_dir":
            sys.modules[module_name] = module_bak


def setup_imports():
    from src.common.registry import registry

    # First, check if imports are already setup
    has_already_setup = registry.get("imports_setup", no_warning=True)
    if has_already_setup:
        return
    # Automatically load all of the modules, so that
    # they register with registry
    root_folder = registry.get("mmf_root", no_warning=True)

    if root_folder is None:
        root_folder = os.path.dirname(os.path.abspath(__file__))
        root_folder = os.path.join(root_folder, "..")

        environment_mmf_path = os.environ.get("MMF_PATH", os.environ.get("PYTHIA_PATH"))

        if environment_mmf_path is not None:
            root_folder = environment_mmf_path

        registry.register("pythia_path", root_folder)
        registry.register("mmf_path", root_folder)

    trainer_folder = os.path.join(root_folder, "trainers")
    trainer_pattern = os.path.join(trainer_folder, "**", "*.py")
    datasets_folder = os.path.join(root_folder, "datasets")
    datasets_pattern = os.path.join(datasets_folder, "**", "*.py")
    model_folder = os.path.join(root_folder, "models")
    model_pattern = os.path.join(model_folder, "**", "*.py")

    importlib.import_module("src.common.meter")

    files = (
        glob.glob(datasets_pattern, recursive=True)
        + glob.glob(model_pattern, recursive=True)
        + glob.glob(trainer_pattern, recursive=True)
    )

    for f in files:
        f = os.path.realpath(f)
        if f.endswith(".py") and not f.endswith("__init__.py"):
            splits = f.split(os.sep)
            import_prefix_index = 0
            for idx, split in enumerate(splits):
                if split == "src":
                    import_prefix_index = idx + 1
            file_name = splits[-1]
            module_name = file_name[: file_name.find(".py")]
            module = ".".join(["src"] + splits[import_prefix_index:-1] + [module_name])
            importlib.import_module(module)

    registry.register("imports_setup", True)
