"""
Load PyTorch Lightning Loggers.
"""

from typing import Union

from omegaconf import DictConfig, ListConfig
from pytorch_lightning.loggers import LightningLoggerBase

from .sigopt import SigoptLogger


class UnknownLogger(Exception):
    """
    This is raised when the logger option in config.yaml file is set incorrectly.
    SigOpt is the only available logger for now.
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
    else:
        raise UnknownLogger(
            f"Unknown logger type: {config.project.logger}. Available loggers are: sigopt.\n"
            f"To enable the logger, set `project.logger` to `true` or `sigopt` in config.yaml\n"
            f"To disable the logger, set `project.logger` to `false`."
        )

    return logger
