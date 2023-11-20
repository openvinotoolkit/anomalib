"""Common helpers for both nightly and pre-merge model tests."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from lightning.pytorch import LightningDataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, ListConfig

from anomalib.config import _update_nncf_config
from anomalib.data import get_datamodule
from anomalib.engine import Engine
from anomalib.models import get_model
from anomalib.models.components import AnomalyModule
from anomalib.utils.callbacks import get_callbacks
from anomalib.utils.callbacks.visualizer import BaseVisualizerCallback
from anomalib.utils.types import TaskType


def model_load_test(config: Union[DictConfig, ListConfig], datamodule: LightningDataModule, results: Dict):
    """Create a new model based on the weights specified in config.

    Args:
        config ([Union[DictConfig, ListConfig]): Model config.
        datamodule (LightningDataModule): Dataloader
        results (Dict): Results from original model.

    """
    loaded_model = get_model(config)  # get new model

    ckpt_path = str(Path(config.trainer.default_root_dir) / "weights" / "last.ckpt")

    callbacks = get_callbacks(config)

    for index, callback in enumerate(callbacks):
        # Remove visualizer callback as saving results takes time
        if isinstance(callback, BaseVisualizerCallback):
            callbacks.pop(index)
            break

    # create new engine object with LoadModel callback (assumes it is present)
    engine = Engine(
        callbacks=callbacks,
        normalization=config.normalization.normalization_method,
        threshold=config.metrics.threshold,
        task=config.task,
        image_metrics=config.metrics.get("image", None),
        pixel_metrics=config.metrics.get("pixel", None),
        **config.trainer,
    )
    # Assumes the new model has LoadModel callback and the old one had ModelCheckpoint callback
    new_results = engine.test(model=loaded_model, datamodule=datamodule, ckpt_path=ckpt_path)[0]
    assert np.isclose(
        results["image_AUROC"], new_results["image_AUROC"]
    ), f"Loaded model does not yield close performance results. {results['image_AUROC']} : {new_results['image_AUROC']}"
    if config.task == "segmentation":
        assert np.isclose(
            results["pixel_AUROC"], new_results["pixel_AUROC"]
        ), "Loaded model does not yield close performance results"
