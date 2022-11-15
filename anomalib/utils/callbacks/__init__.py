"""Callbacks for Anomalib models."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import warnings
from importlib import import_module
from typing import Dict, List, Union

import yaml
from jsonargparse.namespace import Namespace
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import Callback

from anomalib.config import get_default_root_directory
from anomalib.deploy import ExportMode
from anomalib.post_processing import NormalizationMethod

from .cdf_normalization import CdfNormalizationCallback
from .graph import GraphLogger
from .metrics_configuration import MetricsConfigurationCallback
from .min_max_normalization import MinMaxNormalizationCallback
from .model_loader import LoadModelCallback
from .post_processing_configuration import PostProcessingConfigurationCallback
from .tiler_configuration import TilerConfigurationCallback
from .timer import TimerCallback
from .visualizer import ImageVisualizerCallback, MetricVisualizerCallback

__all__ = [
    "CdfNormalizationCallback",
    "GraphLogger",
    "ImageVisualizerCallback",
    "LoadModelCallback",
    "MetricsConfigurationCallback",
    "MetricVisualizerCallback",
    "MinMaxNormalizationCallback",
    "PostProcessingConfigurationCallback",
    "TilerConfigurationCallback",
    "TimerCallback",
]


logger = logging.getLogger(__name__)


def get_callbacks(config: Union[ListConfig, DictConfig]) -> List[Dict]:
    """Return base callbacks for all the lightning models.

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

    callbacks: Dict[str, Dict] = {}

    # Convert trainer callbacks to a dictionary. It makes it easier to search and update values
    # {"anomalib.utils.callbacks.ImageVisualizerCallback":{'task':...}}
    if config.trainer.callbacks is not None:
        for callback in config.trainer.callbacks:
            callbacks[callback.class_path.split(".")[-1]] = dict(callback.init_args)

    monitor = callbacks.get("EarlyStopping", {}).get("monitor", None)
    mode = callbacks.get("EarlyStopping", {}).get("mode", "max")

    if config.trainer.default_root_dir is None:
        config.trainer.default_root_dir = str(get_default_root_directory(config))

    callbacks["pytorch_lightning.callbacks.ModelCheckpoint"] = {
        "dirpath": os.path.join(config.trainer.default_root_dir, "weights"),
        "filename": "model",
        "monitor": monitor,
        "mode": mode,
        "auto_insert_metric_name": False,
    }

    callbacks["TimerCallback"] = {}
    callbacks["PostProcessingConfigurationCallback"] = config.post_processing
    # Add metric configuration to the model via MetricsConfigurationCallback
    callbacks["MetricsConfigurationCallback"] = {"task": config.data.init_args.task, **config.metrics}

    if "resume_from_checkpoint" in config.trainer.keys() and config.trainer.resume_from_checkpoint is not None:
        callbacks["LoadModelCallback"] = {"weights_path": config.trainer.resume_from_checkpoint}

    if config.post_processing.normalization_method is not None:
        # CLI returns an Enum whereas the entrypoint files returns a string
        if isinstance(config.post_processing.normalization_method, NormalizationMethod):
            normalization_method = config.post_processing.normalization_method.name
        else:
            normalization_method = config.post_processing.normalization_method

        if normalization_method == NormalizationMethod.CDF.name:
            if config.model.name in ["padim", "stfpm"]:
                if "nncf" in config.optimization and config.optimization.nncf.apply:
                    raise NotImplementedError("CDF Score Normalization is currently not compatible with NNCF.")
                callbacks["CdfNormalizationCallback"] = {}
            else:
                raise NotImplementedError("Score Normalization is currently supported for PADIM and STFPM only.")
        elif normalization_method == NormalizationMethod.MIN_MAX.name:
            callbacks["MinMaxNormalizationCallback"] = {}
        else:
            raise ValueError(f"Normalization method not recognized: {config.post_processing.normalization_method}")

    add_visualizer_callback(callbacks, config)

    if "optimization" in config.keys():
        if "nncf" in config.optimization and config.optimization.nncf.apply:
            # NNCF wraps torch's jit which conflicts with kornia's jit calls.
            # Hence, nncf is imported only when required
            nncf_config = yaml.safe_load(OmegaConf.to_yaml(config.optimization.nncf))
            callbacks["anomalib.utils.callbacks.nncf.callback.NNCFCallback"] = {
                "config": nncf_config,
                "export_dir": os.path.join(config.trainer.default_root_dir, "compressed"),
            }
        if config.optimization.export_mode is not None:
            logger.info("Setting model export to %s", config.optimization.export_mode)
            callbacks["ExportCallback"] = {
                "input_size": config.model.input_size,
                "dirpath": config.trainer.default_root_dir,
                "filename": "model",
                "export_mode": ExportMode(config.optimization.export_mode),
            }
        else:
            warnings.warn(f"Export option: {config.optimization.export_mode} not found. Defaulting to no model export")

    # Add callback to log graph to loggers
    if config.logging.log_graph not in [None, False]:
        callbacks["GraphLogger"] = {}

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
        instantiated_callbacks.append(callback_class(**init_args))

    return instantiated_callbacks


def add_visualizer_callback(callbacks: Dict[str, Dict], config: Union[DictConfig, ListConfig]):
    """Configure the visualizer callback based on the config and add it to the list of callbacks.

    Args:
        callbacks (Dict[str, Dict]): Current list of callbacks.
        config (Union[DictConfig, ListConfig]): The config object.
    """
    # visualization settings
    assert isinstance(config, (DictConfig, Namespace))
    config.visualization.task = (
        config.data.init_args.task if config.visualization.task is None else config.visualization.task
    )
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
