"""Load PyTorch Lightning Loggers."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import logging
import os
from typing import Iterable, List, Union

from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from pytorch_lightning.loggers import CSVLogger, LightningLoggerBase

from .tensorboard import AnomalibTensorBoardLogger
from .wandb import AnomalibWandbLogger

__all__ = [
    "AnomalibTensorBoardLogger",
    "AnomalibWandbLogger",
    "configure_logger",
    "get_experiment_logger",
]
AVAILABLE_LOGGERS = ["tensorboard", "wandb", "csv"]


class UnknownLogger(Exception):
    """This is raised when the logger option in `config.yaml` file is set incorrectly."""


def configure_logger(level: Union[int, str] = logging.INFO):
    """Get console logger by name.

    Args:
        level (Union[int, str], optional): Logger Level. Defaults to logging.INFO.

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
    config: Union[DictConfig, ListConfig]
) -> Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool]:
    """Return a logger based on the choice of logger in the config file.

    Args:
        config (DictConfig): config.yaml file for the corresponding anomalib model.

    Raises:
        ValueError: for any logger types apart from false and tensorboard

    Returns:
        Union[LightningLoggerBase, Iterable[LightningLoggerBase], bool]: Logger
    """
    if config.project.logger in [None, False]:
        return False

    logger_list: List[LightningLoggerBase] = []
    if isinstance(config.project.logger, str):
        config.project.logger = [config.project.logger]

    for logger in config.project.logger:
        if logger == "tensorboard":
            logger_list.append(
                AnomalibTensorBoardLogger(
                    name="Tensorboard Logs",
                    save_dir=os.path.join(config.project.path, "logs"),
                )
            )
        elif logger == "wandb":
            wandb_logdir = os.path.join(config.project.path, "logs")
            os.makedirs(wandb_logdir, exist_ok=True)
            logger_list.append(
                AnomalibWandbLogger(
                    project=config.dataset.name,
                    name=f"{config.dataset.category} {config.model.name}",
                    save_dir=wandb_logdir,
                )
            )
        elif logger == "csv":
            logger_list.append(CSVLogger(save_dir=os.path.join(config.project.path, "logs")))
        else:
            raise UnknownLogger(
                f"Unknown logger type: {config.project.logger}. "
                f"Available loggers are: {AVAILABLE_LOGGERS}.\n"
                f"To enable the logger, set `project.logger` to `true` or use one of available loggers in config.yaml\n"
                f"To disable the logger, set `project.logger` to `false`."
            )

    return logger_list
