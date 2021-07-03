"""
Load PyTorch Lightning Loggers.
"""

from typing import Union

from omegaconf import DictConfig, ListConfig
from pytorch_lightning.loggers import LightningLoggerBase

from .sigopt import SigoptLogger


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
        raise ValueError("Unknown logger type. Available loggers: false, sigopt, tensorboard")

    return logger
