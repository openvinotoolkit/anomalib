"""Callbacks for Anomalib models."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging
from importlib import import_module
from pathlib import Path

import yaml
from jsonargparse import Namespace
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from omegaconf import DictConfig, ListConfig, OmegaConf

from anomalib.deploy import ExportMode

from .graph import GraphLogger
from .model_loader import LoadModelCallback
from .tiler_configuration import TilerConfigurationCallback
from .timer import TimerCallback
from .visualizer import ImageVisualizerCallback, MetricVisualizerCallback, get_visualization_callbacks

__all__ = [
    "get_visualization_callbacks",
    "GraphLogger",
    "ImageVisualizerCallback",
    "LoadModelCallback",
    "MetricVisualizerCallback",
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

    monitor_metric = (
        None if "early_stopping" not in config.model.init_args else config.model.init_args.early_stopping.metric
    )
    monitor_mode = "max" if "early_stopping" not in config.model.init_args else config.model.early_stopping.mode

    checkpoint = ModelCheckpoint(
        dirpath=Path(config.trainer.default_root_dir) / "weights" / "lightning",
        filename="model",
        monitor=monitor_metric,
        mode=monitor_mode,
        auto_insert_metric_name=False,
    )

    callbacks.extend([checkpoint, TimerCallback()])

    if "ckpt_path" in config.trainer and config.ckpt_path is not None:
        load_model = LoadModelCallback(config.ckpt_path)
        callbacks.append(load_model)

    if "optimization" in config:
        if "nncf" in config.optimization and config.optimization.nncf.apply:
            # NNCF wraps torch's jit which conflicts with kornia's jit calls.
            # Hence, nncf is imported only when required
            nncf_module = import_module("anomalib.utils.callbacks.nncf.callback")
            nncf_callback = nncf_module.NNCFCallback
            nncf_config = yaml.safe_load(OmegaConf.to_yaml(config.optimization.nncf))
            callbacks.append(
                nncf_callback(
                    config=nncf_config,
                    export_dir=str(Path(config.project.path) / "compressed"),
                ),
            )
        if config.optimization.export_mode is not None:
            from .export import ExportCallback

            logger.info("Setting model export to %s", config.optimization.export_mode)
            callbacks.append(
                ExportCallback(
                    input_size=config.model.init_args.input_size,
                    dirpath=config.trainer.default_root_dir,
                    filename="model",
                    export_mode=ExportMode(config.optimization.export_mode),
                ),
            )
        else:
            msg = f"Export option: {config.optimization.export_mode} not found. Defaulting to no model export"
            logger.warning(msg)

    # Add callback to log graph to loggers
    # TODO(ashwinvaidya17): Find location for log_graph key
    # CVS-122658
    # if config.logging.log_graph not in (None, False):

    return callbacks
