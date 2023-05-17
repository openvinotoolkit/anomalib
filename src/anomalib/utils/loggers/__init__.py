"""Load PyTorch Lightning Loggers."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging
from typing import Iterable

from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from pytorch_lightning.loggers import Logger

from .comet import AnomalibCometLogger
from .file_system import FileSystemLogger
from .tensorboard import AnomalibTensorBoardLogger
from .wandb import AnomalibWandbLogger

__all__ = [
    "AnomalibCometLogger",
    "AnomalibTensorBoardLogger",
    "AnomalibWandbLogger",
    "FileSystemLogger",
    "configure_logger",
    "get_experiment_logger",
]


AVAILABLE_LOGGERS = ["tensorboard", "wandb", "file_system", "comet"]


logger = logging.getLogger(__name__)


class UnknownLogger(Exception):
    """This is raised when the logger option in `config.yaml` file is set incorrectly."""


def configure_logger(level: int | str = logging.INFO) -> None:
    """Get console logger by name.

    Args:
        level (int | str, optional): Logger Level. Defaults to logging.INFO.

    Returns:
        Logger: The expected logger.
    """

    if isinstance(level, str):
        level = logging.getLevelName(level)

    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(format=format_string, level=level)

    # Set Pytorch Lightning logs to have a the consistent formatting with anomalib.
    for handler in logging.getLogger("pytorch_lightning").handlers:
        handler.setFormatter(logging.Formatter(format_string))
        handler.setLevel(level)


def get_experiment_logger(
    config: DictConfig | ListConfig,
) -> Logger | Iterable[Logger] | bool:
    """Return a logger based on the choice of logger in the config file.

    Args:
        config (DictConfig): config.yaml file for the corresponding anomalib model.

    Raises:
        ValueError: for any logger types apart from false and tensorboard

    Returns:
        Logger | Iterable[Logger] | bool]: Logger
    """
    logger.info("Loading the experiment logger(s)")

    if config.logging.loggers is None:
        return False

    logger_list: list[Logger] = []
    if not isinstance(config.logging.loggers, (list, ListConfig)):
        config.logging.loggers = [config.logging.loggers]

    for experiment_logger in config.logging.loggers:
        if "tensorboard" in experiment_logger.class_path.lower():
            logger_list.append(AnomalibTensorBoardLogger(**experiment_logger.init_args))
        elif "wandb" in experiment_logger.class_path.lower():
            logger_list.append(AnomalibWandbLogger(**experiment_logger.init_args))
        elif "comet" in experiment_logger.class_path.lower():
            logger_list.append(AnomalibCometLogger(**experiment_logger.init_args))
        elif "filesystem" in experiment_logger.class_path.lower():
            logger_list.append(FileSystemLogger(**experiment_logger.init_args))
        else:
            raise UnknownLogger(
                f"Unknown logger type: {config.logging.loggers}. "
                f"Available loggers are: {AVAILABLE_LOGGERS}.\n"
                f"To enable the logger, set `project.logger` to `true` or use one of available loggers in config.yaml\n"
                f"To disable the logger, set `project.logger` to `false`."
            )

    return logger_list
