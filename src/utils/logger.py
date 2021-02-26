import os
import functools
import logging
import sys
from src.utils.distributed import get_rank
from src.common.registry import registry
from src.utils.configuration import get_mmf_env
from src.utils.file_io import PathManager
from src.utils.timer import Timer

@functools.lru_cache()
def setup_output_folder(folder_only: bool = False):
    """Sets up and returns the output file where the logs will be placed
    based on the configuration passed. Usually "save_dir/logs/log_<timestamp>.txt".
    If env.log_dir is passed, logs will be directly saved in this folder.

    Args:
        folder_only (bool, optional): If folder should be returned and not the file.
            Defaults to False.

    Returns:
        str: folder or file path depending on folder_only flag
    """
    save_dir = get_mmf_env(key="save_dir")
    time_format = "%Y_%m_%dT%H_%M_%S"
    log_filename = "train_"
    log_filename += Timer().get_time_hhmmss(None, format=time_format)
    log_filename += ".log"

    log_folder = os.path.join(save_dir, "logs")

    env_log_dir = get_mmf_env(key="log_dir")
    if env_log_dir:
        log_folder = env_log_dir

    if not PathManager.exists(log_folder):
        PathManager.mkdirs(log_folder)

    if folder_only:
        return log_folder

    log_filename = os.path.join(log_folder, log_filename)

    return log_filename


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return PathManager.open(filename, "a")


# so that calling setup_logger multiple times won't add many handlers
@functools.lru_cache()
def setup_logger(
    output: str = None,
    color: bool = True,
    name: str = "mmf",
    disable: bool = False,
    clear_handlers=True,
    *args,
    **kwargs,
):
    """
    Initialize the MMF logger and set its verbosity level to "INFO".
    Outside libraries shouldn't call this in case they have set there
    own logging handlers and setup. If they do, and don't want to
    clear handlers, pass clear_handlers options.

    The initial version of this function was taken from D2 and adapted
    for MMF.

    Args:
        output (str): a file name or a directory to save log.
            If ends with ".txt" or ".log", assumed to be a file name.
            Default: Saved to file <save_dir/logs/log_[timestamp].txt>
        color (bool): If false, won't log colored logs. Default: true
        name (str): the root module name of this logger. Defaults to "mmf".
        clear_handlers (bool): If false, won't clear existing handlers.

    Returns:
        logging.Logger: a logger
    """
    if disable:
        return None
    logger = logging.getLogger(name)
    logger.propagate = False

    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s : %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    distributed_rank = get_rank()
    handlers = []

    if distributed_rank == 0:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        warnings_logger.addHandler(ch)
        handlers.append(ch)

    # file logging: all workers
    if output is None:
        output = setup_output_folder()

    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "train.log")
        if distributed_rank > 0:
            filename = filename + f".rank{distributed_rank}"
        PathManager.mkdirs(os.path.dirname(filename))

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        warnings_logger.addHandler(fh)
        handlers.append(fh)

        # Slurm/FB output, only log the main process
        if "train.log" not in filename and distributed_rank == 0:
            save_dir = get_mmf_env(key="save_dir")
            filename = os.path.join(save_dir, "train.log")
            sh = logging.StreamHandler(_cached_log_stream(filename))
            sh.setLevel(logging.INFO)
            sh.setFormatter(formatter)
            logger.addHandler(sh)
            warnings_logger.addHandler(sh)
            handlers.append(sh)

        logger.info(f"Logging to: {filename}")

    # Remove existing handlers to add MMF specific handlers
    if clear_handlers:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
    # Now, add our handlers.
    logging.basicConfig(level=logging.INFO, handlers=handlers)

    registry.register("writer", logger)

    return logger


def setup_very_basic_config(color=True):
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s : %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    # Setup a minimal configuration for logging in case something tries to
    # log a message even before logging is setup by MMF.
    logging.basicConfig(level=logging.INFO, handlers=[ch])
