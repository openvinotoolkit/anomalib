"""Callbacks for Anomalib models."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import warnings
from importlib import import_module
from typing import Any, Dict, List, Union

from jsonargparse.namespace import Namespace
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import Callback

from anomalib.post_processing.normalization import NormalizationMethod

from .cdf_normalization import CdfNormalizationCallback
from .graph import GraphLogger
from .metrics_configuration import MetricsConfigurationCallback
from .min_max_normalization import MinMaxNormalizationCallback
from .post_processing_configuration import PostProcessingConfigurationCallback
from .tiler_configuration import TilerConfigurationCallback
from .timer import TimerCallback
from .visualizer import ImageVisualizerCallback, MetricVisualizerCallback

__all__ = [
    "CdfNormalizationCallback",
    "GraphLogger",
    "ImageVisualizerCallback",
    "MetricsConfigurationCallback",
    "MetricVisualizerCallback",
    "MinMaxNormalizationCallback",
    "PostProcessingConfigurationCallback",
    "TilerConfigurationCallback",
    "TimerCallback",
]


logger = logging.getLogger(__name__)


def __update_callback(callbacks: Dict, callback_name: str, update_dict: Dict[str, Any]) -> None:
    """Updates the callback with the given dictionary.

    If the callback exists in the callbacks dictionary, it will be updated with the passed keys.
    This ensure that the callback is not overwritten.
    """
    if callback_name in callbacks:
        # TODO check if keys have been replaced from cli and only then update the keys.
        # basically compare against default keys first
        callbacks[callback_name].update(update_dict)
    else:
        callbacks[callback_name] = update_dict


def get_callbacks(config: Union[ListConfig, DictConfig]) -> List[Callback]:
    """Returns a list of callback objects.

    Args:
        config (DictConfig): Model config

    Return:
        (List[Callback]): List of callbacks
    """
    callbacks_dict = get_callbacks_dict(config)
    callbacks = instantiate_callbacks(callbacks_dict)
    # Remove callbacks from trainer as it is passed separately
    if "callbacks" in config.trainer:
        config.trainer.pop("callbacks")
    return callbacks


def get_callbacks_dict(config: Union[ListConfig, DictConfig]) -> List[Dict]:
    """Returns a list for populating the CLI.

    Args:
        config (DictConfig): Model config

    Return:
        (List[Tuple[str, Dict]]): List of callbacks and init arguments.
          Example:[
            {"class_path": "ModelCheckpoint", "init_args": {...}},
            {"class_path": "PostProcessingConfigurationCallback", "init_args": {...}}
            ]
    """
    logger.info("Loading the callbacks")

    callbacks: Dict = {}

    # Convert trainer callbacks to a dictionary. It makes it easier to search and update values
    # {"anomalib.utils.callbacks.ImageVisualizerCallback":{'task':...}}
    if "callbacks" in config.trainer:
        for callback in config.trainer.callbacks:
            callbacks[callback.class_path.split(".")[-1]] = dict(callback.init_args)

    monitor = callbacks.get("EarlyStopping", {}).get("monitor", None)
    mode = callbacks.get("EarlyStopping", {}).get("mode", "max")

    __update_callback(
        callbacks,
        "ModelCheckpoint",
        {
            "dirpath": os.path.join(config.trainer.default_root_dir, "weights"),
            "filename": "model",
            "monitor": monitor,
            "mode": mode,
            "auto_insert_metric_name": False,
        },
    )

    __update_callback(
        callbacks,
        "PostProcessingConfigurationCallback",
        config.post_processing,
    )

    # Add metric configuration to the model via MetricsConfigurationCallback
    __update_callback(
        callbacks,
        "MetricsConfigurationCallback",
        {
            "task": config.data.init_args.task,
            "image_metrics": config.metrics.get("image_metrics", None),
            "pixel_metrics": config.metrics.get("pixel_metrics", None),
        },
    )

    # Add timing to the pipeline.
    __update_callback(callbacks, "TimerCallback", {})

    #  TODO: This could be set in PostProcessingConfiguration callback
    #   - https://github.com/openvinotoolkit/anomalib/issues/384
    # Normalization.
    normalization = config.post_processing.normalization_method
    if isinstance(normalization, str):
        normalization = NormalizationMethod(normalization.lower())

    if normalization:
        if normalization == NormalizationMethod.MIN_MAX:
            __update_callback(callbacks, "MinMaxNormalizationCallback", {})
        elif normalization == NormalizationMethod.CDF:
            __update_callback(callbacks, "CDFNormalizationCallback", {})
        else:
            raise ValueError(
                f"Unknown normalization type {normalization}. \n"
                f"Available types are either {[member.name for member in NormalizationMethod]}"
            )

    add_visualizer_callback(callbacks, config)

    # Export to OpenVINO
    if "format" in config is not None:
        logger.info("Setting model export to %s", config.format)
        __update_callback(
            callbacks,
            "ExportCallback",
            {
                "input_size": config.data.init_args.image_size,
                "dirpath": os.path.join(config.trainer.default_root_dir, "compressed"),
                "filename": "model",
                "export_mode": config.format,
            },
        )

    if "nncf" in config:
        if os.path.isfile(config.nncf) and config.nncf.endswith(".yaml"):
            __update_callback(
                callbacks,
                "anomalib.core.callbacks.nncf_callback.NNCFCallback",
                {
                    "config": OmegaConf.load(config.nncf),
                    "dirpath": os.path.join(config.trainer.default_root_dir, "compressed"),
                    "filename": "model",
                },
            )
        else:
            raise ValueError(f"--nncf expects a path to nncf config which is a yaml file, but got {config.nncf}")

    # Convert callbacks to dict format expected by pytorch-lightning.
    # eg [
    # {"class_path": "ModelCheckpoint", "init_args": {...}},
    # {"class_path": "PostProcessingConfigurationCallback", "init_args": {...}}
    # ]
    trainer_callbacks = []
    for class_path, init_args in callbacks.items():
        trainer_callbacks.append({"class_path": class_path, "init_args": init_args})

    return trainer_callbacks


def instantiate_callbacks(callbacks: List[Dict]) -> List[Callback]:
    """Instantiate callbacks.

    Args:
        callbacks (List[Dict]): List of callbacks and init arguments.
          Example:[
            {"class_path": "ModelCheckpoint", "init_args": {...}},
            {"class_path": "PostProcessingConfigurationCallback", "init_args": {...}}
            ]

    Return:
        (List[Callback]): List of instantiated callbacks.
    """
    pytorch_callback_module = import_module("pytorch_lightning.callbacks")
    anomalib_callback_module = import_module("anomalib.utils.callbacks")
    instantiated_callbacks = []
    for callback in callbacks:
        class_path = callback["class_path"]
        init_args = callback["init_args"]
        if hasattr(pytorch_callback_module, class_path):
            callback_class = getattr(pytorch_callback_module, class_path)
        elif hasattr(anomalib_callback_module, class_path):
            callback_class = getattr(anomalib_callback_module, class_path)
        elif len(class_path.split(".")) > 1:
            module = import_module(".".join(class_path.split(".")[:-1]))
            callback_class = getattr(module, class_path.split(".")[-1])
        else:
            raise ValueError(f"Callback {class_path} not found.")

        try:
            instantiated_callbacks.append(callback_class(**init_args))
        except Exception as exception:
            raise AttributeError(
                f"Could not instantiate callback {class_path} with arguments {init_args}"
            ) from exception

    return instantiated_callbacks


def add_visualizer_callback(callbacks: Dict[str, Dict], config: Union[DictConfig, ListConfig]):
    """Configure the visualizer callback based on the config and add it to the list of callbacks.

    Args:
        callbacks (Dict[str, Dict]): Current list of callbacks.
        config (Union[DictConfig, ListConfig]): The config object.
    """
    # visualization settings
    assert isinstance(config, (DictConfig, Namespace))
    if "task" in config.visualization:
        warnings.warn(
            "task type should not be configured from visualization explicitly. User data.init_args.task instead."
            f" Setting visualization task to {config.data.init_args.task}."
        )
    config.visualization.task = config.data.init_args.task

    config.visualization.inputs_are_normalized = config.post_processing.normalization_method is not None

    if config.visualization.log_images or config.visualization.save_images or config.visualization.show_images:
        image_save_path = (
            config.visualization.image_save_path
            if config.visualization.image_save_path
            else config.trainer.default_root_dir + "/images"
        )
        for callback in (ImageVisualizerCallback, MetricVisualizerCallback):
            callbacks[callback.__name__] = {
                "task": config.visualization.task,
                "mode": config.visualization.mode,
                "image_save_path": image_save_path,
                "inputs_are_normalized": config.visualization.inputs_are_normalized,
                "show_images": config.visualization.show_images,
                "log_images": config.visualization.log_images,
                "save_images": config.visualization.save_images,
            }
