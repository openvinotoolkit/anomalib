"""Callbacks for Anomalib models."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import warnings
from importlib import import_module
from typing import List, Union

import yaml
from jsonargparse.namespace import Namespace
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from .cdf_normalization import CdfNormalizationCallback
from .graph import GraphLogger
from .metrics_configuration import MetricsConfigurationCallback
from .min_max_normalization import MinMaxNormalizationCallback
from .model_loader import LoadModelCallback
from .tiler_configuration import TilerConfigurationCallback
from .timer import TimerCallback
from .visualizer import ImageVisualizerCallback, MetricVisualizerCallback

__all__ = [
    "CdfNormalizationCallback",
    "LoadModelCallback",
    "MetricsConfigurationCallback",
    "MinMaxNormalizationCallback",
    "TilerConfigurationCallback",
    "TimerCallback",
    "ImageVisualizerCallback",
    "MetricVisualizerCallback",
]


logger = logging.getLogger(__name__)


def get_callbacks(config: Union[ListConfig, DictConfig]) -> List[Callback]:
    """Return base callbacks for all the lightning models.

    Args:
        config (DictConfig): Model config

    Return:
        (List[Callback]): List of callbacks.
    """
    logger.info("Loading the callbacks")

    callbacks: List[Callback] = []

    monitor_metric = None if "early_stopping" not in config.model.keys() else config.model.early_stopping.metric
    monitor_mode = "max" if "early_stopping" not in config.model.keys() else config.model.early_stopping.mode

    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(config.project.path, "weights"),
        filename="model",
        monitor=monitor_metric,
        mode=monitor_mode,
        auto_insert_metric_name=False,
    )

    callbacks.extend([checkpoint, TimerCallback()])

    # Add metric configuration to the model via MetricsConfigurationCallback
    image_metric_names = config.metrics.image if "image" in config.metrics.keys() else None
    pixel_metric_names = config.metrics.pixel if "pixel" in config.metrics.keys() else None
    image_threshold = (
        config.metrics.threshold.image_default if "image_default" in config.metrics.threshold.keys() else None
    )
    pixel_threshold = (
        config.metrics.threshold.pixel_default if "pixel_default" in config.metrics.threshold.keys() else None
    )
    metrics_callback = MetricsConfigurationCallback(
        config.metrics.threshold.adaptive,
        image_threshold,
        pixel_threshold,
        image_metric_names,
        pixel_metric_names,
    )
    callbacks.append(metrics_callback)

    if "resume_from_checkpoint" in config.trainer.keys() and config.trainer.resume_from_checkpoint is not None:
        load_model = LoadModelCallback(config.trainer.resume_from_checkpoint)
        callbacks.append(load_model)

    if "normalization_method" in config.model.keys() and not config.model.normalization_method == "none":
        if config.model.normalization_method == "cdf":
            if config.model.name in ["padim", "stfpm"]:
                if "nncf" in config.optimization and config.optimization.nncf.apply:
                    raise NotImplementedError("CDF Score Normalization is currently not compatible with NNCF.")
                callbacks.append(CdfNormalizationCallback())
            else:
                raise NotImplementedError("Score Normalization is currently supported for PADIM and STFPM only.")
        elif config.model.normalization_method == "min_max":
            callbacks.append(MinMaxNormalizationCallback())
        else:
            raise ValueError(f"Normalization method not recognized: {config.model.normalization_method}")

    add_visualizer_callback(callbacks, config)

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
        if "openvino" in config.optimization and config.optimization.openvino.apply:
            from .openvino import (  # pylint: disable=import-outside-toplevel
                OpenVINOCallback,
            )

            callbacks.append(
                OpenVINOCallback(
                    input_size=config.model.input_size,
                    dirpath=os.path.join(config.project.path, "openvino"),
                    filename="model",
                )
            )

    # Add callback to log graph to loggers
    if config.logging.log_graph not in [None, False]:
        callbacks.append(GraphLogger())

    return callbacks


def add_visualizer_callback(callbacks: List[Callback], config: Union[DictConfig, ListConfig]):
    """Configure the visualizer callback based on the config and add it to the list of callbacks.

    Args:
        callbacks (List[Callback]): Current list of callbacks.
        config (Union[DictConfig, ListConfig]): The config object.
    """
    # visualization settings
    assert isinstance(config, (DictConfig, Namespace))
    # TODO remove this when version is upgraded to 0.4.0
    if isinstance(config, DictConfig):
        if (
            "log_images_to" in config.project.keys()
            and len(config.project.log_images_to) > 0
            or "log_images_to" in config.logging.keys()
            and len(config.logging.log_images_to) > 0
        ):
            warnings.warn(
                "log_images_to parameter is deprecated and will be removed in version 0.4.0 Please use "
                "the visualization.log_images and visualization.save_images parameters instead."
            )
            if "visualization" not in config.keys():
                config["visualization"] = dict(
                    log_images=False, save_images=False, show_image=False, image_save_path=None
                )
            if "local" in config.project.log_images_to:
                config.visualization["save_images"] = True
            if "local" not in config.project.log_images_to or len(config.project.log_images_to) > 1:
                config.visualization["log_images"] = True
        config.visualization.task = config.dataset.task
        config.visualization.inputs_are_normalized = not config.model.normalization_method == "none"
    else:
        config.visualization.task = config.data.init_args.task
        config.visualization.inputs_are_normalized = not config.metrics.normalization_method == "none"
    if config.visualization.log_images or config.visualization.save_images or config.visualization.show_images:
        image_save_path = (
            config.visualization.image_save_path
            if config.visualization.image_save_path
            else config.project.path + "/images"
        )
        for callback in (ImageVisualizerCallback, MetricVisualizerCallback):
            callbacks.append(
                callback(
                    task=config.visualization.task,
                    mode=config.visualization.mode,
                    image_save_path=image_save_path,
                    inputs_are_normalized=config.visualization.inputs_are_normalized,
                    show_images=config.visualization.show_images,
                    log_images=config.visualization.log_images,
                    save_images=config.visualization.save_images,
                )
            )
