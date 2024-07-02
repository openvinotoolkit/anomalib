"""Load PyTorch Lightning Loggers."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from rich.logging import RichHandler

__all__ = [
    "configure_logger",
    "get_experiment_logger",
]

try:
    from .comet import AnomalibCometLogger  # noqa: F401
    from .mlflow import AnomalibMLFlowLogger  # noqa: F401
    from .tensorboard import AnomalibTensorBoardLogger  # noqa: F401
    from .wandb import AnomalibWandbLogger  # noqa: F401

    __all__.extend(
        [
            "AnomalibCometLogger",
            "AnomalibTensorBoardLogger",
            "AnomalibWandbLogger",
            "AnomalibMLFlowLogger",
        ],
    )
except ImportError:
    print("To use any logger install it using `anomalib install -v`")


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
