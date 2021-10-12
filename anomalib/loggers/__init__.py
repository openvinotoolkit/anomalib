"""
Load PyTorch Lightning Loggers.
"""


import os
from typing import Union

from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from pytorch_lightning.loggers.base import LightningLoggerBase

from .sigopt import SigoptLogger
from .tensorboard import AnomalibTensorBoardLogger

__all__ = ["SigoptLogger", "AnomalibTensorBoardLogger", "get_logger"]
AVAILABLE_LOGGERS = ["sigopt", "tensorboard"]


class UnknownLogger(Exception):
    """
    This is raised when the logger option in config.yaml file is set incorrectly.
    """


def get_logger(config: Union[DictConfig, ListConfig]) -> Union[LightningLoggerBase, bool]:
    """
    Return a logger based on the choice of logger in the config file.

    Args:
        config (DictConfig): config.yaml file for the corresponding anomalib model.

    Raises:
        ValueError: for any logger types apart from false, sigopt and tensorboard

    Returns:
        Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool]: Logger
    """

    logger: Union[LightningLoggerBase, bool]

    if config.project.logger in [None, False]:
        logger = False
    elif config.project.logger in ["sigopt", True]:
        logger = SigoptLogger(
            project=config.dataset.name,
            name=f"{config.dataset.category} {config.model.name}",
        )
    elif config.project.logger == "tensorboard":
        logger = AnomalibTensorBoardLogger(
            name="Tensorboard Logs",
            save_dir=os.path.join(config.project.path, "logs"),
        )
    else:
        raise UnknownLogger(
            f"Unknown logger type: {config.project.logger}. "
            f"Available loggers are: {AVAILABLE_LOGGERS}.\n"
            f"To enable the logger, set `project.logger` to `true` or use one of available loggers in config.yaml\n"
            f"To disable the logger, set `project.logger` to `false`."
        )

    return logger
