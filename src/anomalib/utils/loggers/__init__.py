"""Load PyTorch Lightning Loggers."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging
from collections.abc import Iterable
from pathlib import Path

from lightning.pytorch.loggers import CSVLogger, Logger
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from rich.logging import RichHandler

from .comet import AnomalibCometLogger
from .tensorboard import AnomalibTensorBoardLogger
from .wandb import AnomalibWandbLogger

__all__ = [
    "AnomalibCometLogger",
    "AnomalibTensorBoardLogger",
    "AnomalibWandbLogger",
    "configure_logger",
    "get_experiment_logger",
]


AVAILABLE_LOGGERS = ["tensorboard", "wandb", "csv", "comet"]


logger = logging.getLogger(__name__)


class UnknownLoggerError(Exception):
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
    logging.getLogger().addHandler(RichHandler(rich_tracebacks=True))

    # Set Pytorch Lightning logs to have a the consistent formatting with anomalib.
    for handler in logging.getLogger("lightning.pytorch").handlers:
        handler.setFormatter(logging.Formatter(format_string))
        handler.setLevel(level)
    logging.getLogger("lightning.pytorch").addHandler(RichHandler(rich_tracebacks=True))


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

    if "logger" not in config.trainer or config.trainer.logger in (None, False):
        return False

    logger_list: list[Logger] = []
    if isinstance(config.trainer.logger, str):
        config.trainer.logger = [config.trainer.logger]

    for experiment_logger in config.trainer.logger:
        if experiment_logger == "tensorboard":
            logger_list.append(
                AnomalibTensorBoardLogger(
                    name="Tensorboard Logs",
                    save_dir=str(Path(config.project.path) / "logs"),
                    log_graph=False,  # TODO: find location for log_graph key
                ),
            )
        elif experiment_logger == "wandb":
            wandb_logdir = str(Path(config.project.path) / "logs")
            Path(wandb_logdir).mkdir(parents=True, exist_ok=True)
            name = (
                config.model.class_path.split(".")[-1]
                if "category" not in config.data.init_args
                else f"{config.data.init_args.category} {config.model.class_path.split('.')[-1]}"
            )
            logger_list.append(
                AnomalibWandbLogger(
                    project=config.data.class_path.split(".")[-1],
                    name=name,
                    save_dir=wandb_logdir,
                ),
            )
        elif experiment_logger == "comet":
            comet_logdir = str(Path(config.project.path) / "logs")
            Path(comet_logdir).mkdir(parents=True, exist_ok=True)
            run_name = (
                config.model.name
                if "category" not in config.data.init_args
                else f"{config.data.init_args.category} {config.model.class_path.split('.')[-1]}"
            )
            logger_list.append(
                AnomalibCometLogger(
                    project_name=config.data.class_path.split(".")[-1],
                    experiment_name=run_name,
                    save_dir=comet_logdir,
                ),
            )
        elif experiment_logger == "csv":
            logger_list.append(CSVLogger(save_dir=Path(config.project.path) / "logs"))
        else:
            msg = (
                f"Unknown logger type: {config.trainer.logger}. Available loggers are: {AVAILABLE_LOGGERS}.\n"
                "To enable the logger, set `project.logger` to `true` or use one of available loggers in "
                "config.yaml\nTo disable the logger, set `project.logger` to `false`."
            )
            raise UnknownLoggerError(
                msg,
            )

    # TODO remove this method and set these values in ``update_config``
    del config.trainer.logger

    return logger_list
