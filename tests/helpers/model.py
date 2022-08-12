"""Common helpers for both nightly and pre-merge model tests."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from anomalib.config import get_configurable_parameters, update_nncf_config
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.models.components import AnomalyModule
from anomalib.utils.callbacks import get_callbacks
from anomalib.utils.callbacks.visualizer import BaseVisualizerCallback


def setup_model_train(
    model_name: str,
    dataset_path: str,
    project_path: str,
    nncf: bool,
    category: str,
    score_type: Optional[str] = None,
    fast_run: bool = False,
    dataset_task: Optional[str] = None,
    visualizer_mode: Optional[str] = None,
    device: Union[List[int], int] = [0],
) -> Tuple[Union[DictConfig, ListConfig], LightningDataModule, AnomalyModule, Trainer]:
    """Train the model based on the parameters passed.

    Args:
        model_name (str): Name of the model to train.
        dataset_path (str): Location of the dataset.
        project_path (str): Path to temporary project folder.
        nncf (bool): Add nncf callback.
        category (str): Category to train on.
        score_type (str, optional): Only used for DFM. Defaults to None.
        fast_run (bool, optional): If set to true, the model trains for only 1 epoch. We train for one epoch as
            this ensures that both anomalous and non-anomalous images are present in the validation step.
        dataset_task (str, optional): Specify the type of task. Must be in ["classification", "segmentation"].
            Used for integration testing of model / task / visualizer_mode.
        visualizer_mode (str, optional): Specify the type of visualization. Must be in ["full", "simple"].
            Used for integration testing of model / task / visualizer_mode.
        device (List[int], int, optional): Select which device you want to train the model on. Defaults to first GPU.

    Returns:
        Tuple[DictConfig, LightningDataModule, AnomalyModule, Trainer]: config, datamodule, trained model, trainer
    """
    config = get_configurable_parameters(model_name=model_name)
    if score_type is not None:
        config.model.score_type = score_type
    config.project.seed = 42
    config.dataset.category = category
    config.dataset.path = dataset_path
    config.project.log_images_to = []
    config.trainer.devices = device
    config.trainer.accelerator = "gpu" if device != 0 else "cpu"
    if dataset_task is not None:
        config.dataset.task = dataset_task
    if visualizer_mode is not None:
        config.visualization.mode = visualizer_mode
        config.visualization.save_images = True  # Enforce processing by Visualizer
        if "pixel" in config.metrics and dataset_task == "classification":
            del config.metrics.pixel

    # Remove legacy flags
    for legacy_device in ["num_processes", "gpus", "ipus", "tpu_cores"]:
        if legacy_device in config.trainer:
            config.trainer[legacy_device] = None

    if nncf:
        config.optimization["nncf"] = {"apply": True, "input_info": {"sample_size": None}}
        config = update_nncf_config(config)
        config.init_weights = None

    # reassign project path as config is updated in `update_config_for_nncf`
    config.project.path = project_path

    datamodule = get_datamodule(config)
    model = get_model(config)

    callbacks = get_callbacks(config)

    # Force model checkpoint to create checkpoint after first epoch
    if fast_run == True:
        for index, callback in enumerate(callbacks):
            if isinstance(callback, ModelCheckpoint):
                callbacks.pop(index)
                break
        model_checkpoint = ModelCheckpoint(
            dirpath=os.path.join(config.project.path, "weights"),
            filename="last",
            monitor=None,
            mode="max",
            save_last=True,
            auto_insert_metric_name=False,
        )
        callbacks.append(model_checkpoint)

    for index, callback in enumerate(callbacks):
        if isinstance(callback, BaseVisualizerCallback):
            callbacks.pop(index)
            break

    # Train the model.
    if fast_run:
        config.trainer.max_epochs = 1
        config.trainer.check_val_every_n_epoch = 1

    trainer = Trainer(callbacks=callbacks, **config.trainer)
    trainer.fit(model=model, datamodule=datamodule)
    return config, datamodule, model, trainer


def model_load_test(config: Union[DictConfig, ListConfig], datamodule: LightningDataModule, results: Dict):
    """Create a new model based on the weights specified in config.

    Args:
        config ([Union[DictConfig, ListConfig]): Model config.
        datamodule (LightningDataModule): Dataloader
        results (Dict): Results from original model.

    """
    loaded_model = get_model(config)  # get new model
    # Assing the weight file to resume_from_checkpoint. When trainer is initialized, Trainer
    # object will automatically load the weights.
    config.trainer.resume_from_checkpoint = os.path.join(config.project.path, "weights/last.ckpt")

    callbacks = get_callbacks(config)

    for index, callback in enumerate(callbacks):
        # Remove visualizer callback as saving results takes time
        if isinstance(callback, BaseVisualizerCallback):
            callbacks.pop(index)
            break

    # create new trainer object with LoadModel callback (assumes it is present)
    trainer = Trainer(callbacks=callbacks, **config.trainer)
    # Assumes the new model has LoadModel callback and the old one had ModelCheckpoint callback
    new_results = trainer.test(model=loaded_model, datamodule=datamodule)[0]
    assert np.isclose(
        results["image_AUROC"], new_results["image_AUROC"]
    ), f"Loaded model does not yield close performance results. {results['image_AUROC']} : {new_results['image_AUROC']}"
    if config.dataset.task == "segmentation":
        assert np.isclose(
            results["pixel_AUROC"], new_results["pixel_AUROC"]
        ), "Loaded model does not yield close performance results"
