"""Callbacks for Anomalib models."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging
import os
import warnings
from importlib import import_module

import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import Callback

from anomalib import deploy

from .graph import GraphLogger
from .tiler_configuration import TilerConfigurationCallback
from .timer import TimerCallback

__all__ = [
    "GraphLogger",
    "TilerConfigurationCallback",
    "TimerCallback",
]


logger = logging.getLogger(__name__)


def get_callbacks(config: DictConfig | ListConfig) -> list[Callback]:
    """Return base callbacks for all the lightning models.

    Args:
        config (DictConfig): Model config

    Return:
        (list[Callback]): List of callbacks.
    """
    logger.info("Loading the callbacks")

    callbacks: list[Callback] = []

    if "optimization" in config.keys():
        if "nncf" in config.optimization and config.optimization.nncf.apply:
            # NNCF wraps torch's jit which conflicts with kornia's jit calls.
            # Hence, nncf is imported only when required
            nncf_module = import_module("anomalib.utils.callbacks.nncf.callback")
            nncf_callback = getattr(nncf_module, "NNCFCallback")
            nncf_config = yaml.safe_load(OmegaConf.to_yaml(config.optimization.nncf))
            callbacks.append(
                nncf_callback(
                    config=nncf_config,
                    export_dir=os.path.join(config.project.path, "compressed"),
                )
            )
        if config.optimization.export_mode is not None:
            from .export import ExportCallback  # pylint: disable=import-outside-toplevel

            logger.info("Setting model export to %s", config.optimization.export_mode)
            callbacks.append(
                ExportCallback(
                    input_size=config.model.input_size,
                    dirpath=config.project.path,
                    filename="model",
                    export_mode=deploy.ExportMode(config.optimization.export_mode),
                )
            )
        else:
            warnings.warn(f"Export option: {config.optimization.export_mode} not found. Defaulting to no model export")

    # Add callback to log graph to loggers
    if config.logging.log_graph not in (None, False):
        callbacks.append(GraphLogger())

    return callbacks
