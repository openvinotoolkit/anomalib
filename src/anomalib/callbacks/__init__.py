"""Callbacks for Anomalib models."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging

from jsonargparse import Namespace
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig, ListConfig

from .checkpoint import ModelCheckpoint
from .graph import GraphLogger
from .model_loader import LoadModelCallback
from .tiler_configuration import TilerConfigurationCallback
from .timer import TimerCallback

__all__ = [
    "ModelCheckpoint",
    "GraphLogger",
    "LoadModelCallback",
    "TilerConfigurationCallback",
    "TimerCallback",
]


logger = logging.getLogger(__name__)


def get_callbacks(config: DictConfig | ListConfig | Namespace) -> list[Callback]:
    """Return base callbacks for all the lightning models.

    Args:
        config (DictConfig | ListConfig | Namespace): Model config

    Return:
        (list[Callback]): List of callbacks.
    """
    logger.info("Loading the callbacks")

    callbacks: list[Callback] = []

    if "ckpt_path" in config.trainer and config.ckpt_path is not None:
        load_model = LoadModelCallback(config.ckpt_path)
        callbacks.append(load_model)

    return callbacks
