"""Functions used to train and use ensemble of models."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
import os
import warnings
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.tiled_ensemble.ensemble_tiler import EnsembleTiler
from tools.tiled_ensemble.post_processing.postprocess import NormalizationStage
from tools.tiled_ensemble.predictions import (
    BasicEnsemblePredictions,
    EnsemblePredictions,
    FileSystemEnsemblePredictions,
    RescaledEnsemblePredictions,
)

from anomalib.data import get_datamodule
from anomalib.data.base.datamodule import AnomalibDataModule, collate_fn
from anomalib.utils.callbacks import (
    GraphLogger,
    LoadModelCallback,
    MetricsConfigurationCallback,
    MinMaxNormalizationCallback,
    PostProcessingConfigurationCallback,
    TimerCallback,
)

logger = logging.getLogger(__name__)


class TileCollater:
    """
    Class serving as collate function to perform tiling on batch of images from Dataloader.

    Args:
        tiler (EnsembleTiler): Tiler used to split the images to tiles.
        tile_index (tuple[int, int]): Index of tile we want to return.
    """

    def __init__(self, tiler: EnsembleTiler, tile_index: tuple[int, int]) -> None:
        self.tiler = tiler
        self.tile_index = tile_index

    def __call__(self, batch: list) -> dict[str, Any]:
        """
        Collate batch and tile images + masks from batch.

        Args:
            batch (list): Batch of elements from data, also including images.

        Returns:
            dict[str, Any]: Collated batch dictionary with tiled images.
        """
        # use default collate
        coll_batch = collate_fn(batch)

        tiled_images = self.tiler.tile(coll_batch["image"])
        # return only tiles at given index
        coll_batch["image"] = tiled_images[self.tile_index]

        if "mask" in coll_batch.keys():
            # insert channel (as mask has just one)
            tiled_masks = self.tiler.tile(coll_batch["mask"].unsqueeze(1))

            # return only tiled at given index, squeeze to remove previously added channel
            coll_batch["mask"] = tiled_masks[self.tile_index].squeeze(1)

        return coll_batch


def prepare_ensemble_configurable_parameters(
    ens_config_path: str, config: DictConfig | ListConfig
) -> DictConfig | ListConfig:
    """Add all ensemble configuration parameters to config object.

    Args:
        ens_config_path (str): Path to ensemble configuration.
        config (DictConfig | ListConfig): Configurable parameters object.

    Returns:
        DictConfig | ListConfig: Configurable parameters object with ensemble parameters.
    """
    ens_config = OmegaConf.load(ens_config_path)
    config["ensemble"] = ens_config

    config.ensemble.tiling.tile_size = EnsembleTiler.validate_size_type(ens_config.tiling.tile_size)
    # update model input size
    config.model.input_size = config.ensemble.tiling.tile_size

    (Path(config.project.path) / "config_tiled_ensemble.yaml").write_text(OmegaConf.to_yaml(config))

    return config


def get_ensemble_datamodule(config: DictConfig | ListConfig, tiler: EnsembleTiler) -> AnomalibDataModule:
    """
    Get Anomaly Datamodule adjusted for use in ensemble.

    Datamodule collate function gets replaced by TileCollater in order to tile all images before they are passed on.

    Args:
        config (DictConfig | ListConfig): Configuration of the anomaly model.
        tiler (EnsembleTiler): Tiler used to split the images to tiles for use in ensemble.
    Returns:
        AnomalibDataModule: Anomalib Lightning DataModule
    """
    datamodule = get_datamodule(config)
    # set custom collate function that does the tiling
    datamodule.collate_fn = TileCollater(tiler, (0, 0))
    return datamodule


def get_prediction_storage(config: DictConfig | ListConfig) -> tuple[EnsemblePredictions, EnsemblePredictions]:
    """
    Return prediction storage class as set in config.

    Predictions can be stored directly in memory (direct), on file system (file_system)
    or downscaled in memory (rescaled).

    Args:
        config (DictConfig | ListConfig): Configurable parameters object.

    Returns:
        tuple[EnsemblePredictions, EnsemblePredictions]: Storage of ensemble and validation predictions.
    """
    # store predictions in memory
    if config.ensemble.predictions.storage == "direct":
        ensemble_pred = BasicEnsemblePredictions()
    # store predictions on file system
    elif config.ensemble.predictions.storage == "file_system":
        ensemble_pred = FileSystemEnsemblePredictions(storage_path=config.project.path)
    # store downscaled predictions in memory
    elif config.ensemble.predictions.storage == "rescaled":
        ensemble_pred = RescaledEnsemblePredictions(config.ensemble.predictions.rescale_factor)
    else:
        raise ValueError(
            f"Prediction storage not recognized: {config.ensemble.predictions.storage}."
            f" Possible values: [direct, file_system, rescaled]."
        )

    # if val is same as test, don't process twice
    if config.dataset.val_split_mode == "same_as_test":
        validation_pred = ensemble_pred
    else:
        validation_pred = copy.deepcopy(ensemble_pred)

    return ensemble_pred, validation_pred


def get_ensemble_callbacks(config: DictConfig | ListConfig, tile_index: tuple[int, int]) -> list[Callback]:
    """
    Return base callbacks for ensemble.

    Thresholding is added everytime, even if the thresholding is set to final, joined level thresholding
    as it is required by some metrics for early stopping.

    Only valid normalization for ensemble is MinMax that can be either applied on tile level, which means it is
    included as part of this callback. If it's used at the end when images are joined, it's not included in callbacks.

    Ensemble doesn't support nncf optimization, so it can't be added as a callback.

    Args:
        config (DictConfig | ListConfig): Model config file.
        tile_index (tuple[int, int]): Index of current tile in ensemble.

    Return:
        list[Callback]: List of callbacks.
    """
    logger.info("Loading the ensemble callbacks")

    callbacks: list[Callback] = []

    monitor_metric = None if "early_stopping" not in config.model.keys() else config.model.early_stopping.metric
    monitor_mode = "max" if "early_stopping" not in config.model.keys() else config.model.early_stopping.mode

    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(config.project.path, "weights", "lightning"),
        filename=f"model{tile_index[0]}_{tile_index[1]}",
        monitor=monitor_metric,
        mode=monitor_mode,
        auto_insert_metric_name=False,
    )

    callbacks.extend([checkpoint, TimerCallback()])

    if "resume_from_checkpoint" in config.trainer.keys() and config.trainer.resume_from_checkpoint is not None:
        load_model = LoadModelCallback(config.trainer.resume_from_checkpoint)
        callbacks.append(load_model)

    # Add post-processing configurations to AnomalyModule.
    image_threshold = (
        config.ensemble.metrics.threshold.manual_image
        if "manual_image" in config.ensemble.metrics.threshold.keys()
        else None
    )
    pixel_threshold = (
        config.ensemble.metrics.threshold.manual_pixel
        if "manual_pixel" in config.ensemble.metrics.threshold.keys()
        else None
    )

    # even if we threshold at the end, we want to have this here due to some models that need early stopping criteria
    post_processing_callback = PostProcessingConfigurationCallback(
        threshold_method=config.ensemble.metrics.threshold.method,
        manual_image_threshold=image_threshold,
        manual_pixel_threshold=pixel_threshold,
    )
    callbacks.append(post_processing_callback)

    # Add metric configuration to the model via MetricsConfigurationCallback
    metrics_callback = MetricsConfigurationCallback(
        config.dataset.task,
        config.ensemble.metrics.get("image", None),
        config.ensemble.metrics.get("pixel", None),
    )
    callbacks.append(metrics_callback)

    # if we normalize each tile separately
    if config.ensemble.post_processing.normalization == NormalizationStage.INDIVIDUAL_TILE:
        if "normalization_method" in config.model.keys() and not config.model.normalization_method == "none":
            if config.model.normalization_method == "min_max":
                callbacks.append(MinMaxNormalizationCallback())
            else:
                raise ValueError(
                    f"Ensemble only supports MinMax normalization. "
                    f"Normalization method not recognized: {config.model.normalization_method}"
                )

    if "optimization" in config.keys():
        if "nncf" in config.optimization and config.optimization.nncf.apply:
            warnings.warn("NNCF is not supported with ensemble.")

        if config.optimization.export_mode is not None:
            warnings.warn("Exporting tiled ensemble is currently not supported.")

    # Add callback to log graph to loggers
    if config.logging.log_graph not in (None, False):
        callbacks.append(GraphLogger())

    return callbacks
