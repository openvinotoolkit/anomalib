"""Functions used to train and use ensemble of models."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import warnings
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor

from anomalib.data import get_datamodule
from anomalib.data.base.datamodule import collate_fn, AnomalibDataModule
from anomalib.deploy import ExportMode
from anomalib.models.ensemble.ensemble_prediction_data import EnsemblePredictions
from anomalib.models.ensemble.ensemble_prediction_joiner import EnsemblePredictionJoiner
from anomalib.models.ensemble.ensemble_tiler import EnsembleTiler
from anomalib.utils.callbacks import (
    TimerCallback,
    LoadModelCallback,
    PostProcessingConfigurationCallback,
    MinMaxNormalizationCallback,
    GraphLogger,
    MetricsConfigurationCallback,
)
from anomalib.models.ensemble.ensemble_prediction_data import (
    BasicEnsemblePredictions,
    FileSystemEnsemblePredictions,
    RescaledEnsemblePredictions,
)

logger = logging.getLogger(__name__)


class TileCollater:
    """
    Class serving as collate function to perform tiling on batch of images from Dataloader.

    Args:
        tiler: Tiler used to split the images to tiles.
        tile_index: Index of tile we want to return.
    """

    def __init__(self, tiler: EnsembleTiler, tile_index: (int, int)) -> None:
        self.tiler = tiler
        self.tile_index = tile_index

    def __call__(self, batch: list) -> dict[str, Any]:
        """
        Collate batch and tile images + masks from batch.

        Args:
            batch: Batch of elements from data, also including images.

        Returns:
            Collated batch dictionary with tiled images.
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
    ens_config_path, config: DictConfig | ListConfig
) -> DictConfig | ListConfig:
    """
    Add all ensemble configuration parameters to config object

    Args:
        ens_config_path: Path to ensemble configuration.
        config: Configurable parameters object.

    Returns:
        Configurable parameters object with ensemble parameters.
    """
    ens_config = OmegaConf.load(ens_config_path)
    config["ensemble"] = ens_config

    config.ensemble.tiling.tile_size = EnsembleTiler.validate_size_type(ens_config.tiling.tile_size)
    # update model input size
    config.model.input_size = config.ensemble.tiling.tile_size

    return config


def get_ensemble_datamodule(config: DictConfig | ListConfig, tiler: EnsembleTiler) -> AnomalibDataModule:
    """
    Get Anomaly Datamodule adjusted for use in ensemble.

    Args:
        config: Configuration of the anomaly model.
        tiler: Tiler used to split the images to tiles for use in ensemble.
    Returns:
        PyTorch Lightning DataModule
    """
    datamodule = get_datamodule(config)
    datamodule.custom_collate_fn = TileCollater(tiler, (0, 0))
    return datamodule


def get_prediction_storage(config: DictConfig | ListConfig) -> (EnsemblePredictions, EnsemblePredictions):
    if config.ensemble.predictions.storage == "direct":
        return BasicEnsemblePredictions(), BasicEnsemblePredictions()
    elif config.ensemble.predictions.storage == "file_system":
        return FileSystemEnsemblePredictions(config), FileSystemEnsemblePredictions(config)
    elif config.ensemble.predictions.storage == "rescaled":
        return RescaledEnsemblePredictions(config), RescaledEnsemblePredictions(config)
    else:
        raise ValueError(
            f"Prediction storage not recognized: {config.ensemble.predictions.storage}."
            f" Possible values: [direct, file_system, rescaled]."
        )


def get_ensemble_callbacks(config: DictConfig | ListConfig, tile_index: (int, int)) -> list[Callback]:
    """Return base callbacks for ensemble.

    Args:
        config: Model config file.
        tile_index: Index of current tile in ensemble.

    Return:
        List of callbacks.
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
        config.ensemble.metrics.threshold.manual_image if "manual_image" in config.metrics.threshold.keys() else None
    )
    pixel_threshold = (
        config.ensemble.metrics.threshold.manual_pixel if "manual_pixel" in config.metrics.threshold.keys() else None
    )

    # even if we threshold at the end, we want to have this here due to some models that need early stopping criteria
    post_processing_callback = PostProcessingConfigurationCallback(
        threshold_method=config.metrics.threshold.method,
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

    if config.ensemble.post_processing.normalization == "tile":
        if "normalization_method" in config.model.keys() and not config.model.normalization_method == "none":
            if config.model.normalization_method == "min_max":
                callbacks.append(MinMaxNormalizationCallback())
            else:
                raise ValueError(
                    f"Ensemble only supports MinMax normalization. Normalization method not recognized: {config.model.normalization_method}"
                )

    if "optimization" in config.keys():
        if "nncf" in config.optimization and config.optimization.nncf.apply:
            warnings.warn(f"NNCF is not supported with ensemble.")

        if config.optimization.export_mode is not None:
            from src.anomalib.utils.callbacks.export import (  # pylint: disable=import-outside-toplevel
                ExportCallback,
            )

            logger.info("Setting model export to %s", config.optimization.export_mode)
            callbacks.append(
                ExportCallback(
                    input_size=config.model.input_size,
                    dirpath=config.project.path,
                    filename=f"model{tile_index[0]}_{tile_index[1]}",
                    export_mode=ExportMode(config.optimization.export_mode),
                )
            )
        else:
            warnings.warn(f"Export option: {config.optimization.export_mode} not found. Defaulting to no model export")

    # Add callback to log graph to loggers
    if config.logging.log_graph not in (None, False):
        callbacks.append(GraphLogger())

    return callbacks


class BasicPredictionJoiner(EnsemblePredictionJoiner):
    """
    Basic implementation of ensemble prediction joiner.
    Tiles are put together and untiled.
    Boxes are stacked per image.
    Labels are combined with OR operator, meaning one anomalous tile -> anomalous image
    Scores are averaged


    """

    def join_tiles(self, batch_data: dict, tile_key: str) -> Tensor:
        """
        Join tiles back into one tensor and perform untiling with tiler.

        Args:
            batch_data: Dictionary containing all tile predictions of current batch.
            tile_key: Key used in prediction dictionary for tiles that we want to join

        Returns:
            Tensor of tiles in original (stitched) shape.
        """
        # tiles with index (0, 0) always exists
        first_tiles = batch_data[(0, 0)][tile_key]

        # get batch and device from tiles
        batch_size = first_tiles.shape[0]
        device = first_tiles.device

        if tile_key == "mask":
            # in case of ground truth masks, we don't have channels
            joined_size = (
                self.tiler.num_patches_h,
                self.tiler.num_patches_w,
                batch_size,
                self.tiler.tile_size_h,
                self.tiler.tile_size_w,
            )
        else:
            # all tiles beside masks also have channels
            num_channels = first_tiles.shape[1]
            joined_size = (
                self.tiler.num_patches_h,
                self.tiler.num_patches_w,
                batch_size,
                num_channels,
                self.tiler.tile_size_h,
                self.tiler.tile_size_w,
            )

        # create new empty tensor for joined tiles
        joined_masks = torch.zeros(size=joined_size, device=device)

        # insert tile into joined tensor at right locations
        for (tile_i, tile_j), tile_data in batch_data.items():
            joined_masks[tile_i, tile_j, ...] = tile_data[tile_key]

        if tile_key == "mask":
            # add channel as tiler needs it
            joined_masks = joined_masks.unsqueeze(3)

        # stitch tiles back into whole, output is [B, C, H, W]
        joined_output = self.tiler.untile(joined_masks)

        if tile_key == "mask":
            # remove previously added channels
            joined_output = joined_output.squeeze(1)

        return joined_output

    def join_boxes(self, batch_data: dict) -> dict:
        """
        Join boxes data from all tiles. This includes pred_boxes, box_scores and box_labels.
        Joining is done by stacking boxes from all tiles.

        Args:
            batch_data: Dictionary containing all tile predictions of current batch.

        Returns:
            Dictionary with joined boxes, box scores and box labels.
        """
        batch_size = len(batch_data[(0, 0)]["pred_boxes"])

        # create array of placeholder arrays, that will contain all boxes for each image
        boxes = [[] for _ in range(batch_size)]
        scores = [[] for _ in range(batch_size)]
        labels = [[] for _ in range(batch_size)]

        # go over all tiles and add box data tensor to belonging array
        for (tile_i, tile_j), curr_tile_pred in batch_data.items():
            for i in range(batch_size):
                # boxes have form [x_1, y_1, x_2, y_2]
                curr_boxes = curr_tile_pred["pred_boxes"][i]

                # tile position offset
                offset_w = self.tiler.tile_size_w * tile_j
                offset_h = self.tiler.tile_size_h * tile_i

                # offset in x-axis
                curr_boxes[:, 0] += offset_w
                curr_boxes[:, 2] += offset_w

                # offset in y-axis
                curr_boxes[:, 1] += offset_h
                curr_boxes[:, 3] += offset_h

                boxes[i].append(curr_boxes)
                scores[i].append(curr_tile_pred["box_scores"][i])
                labels[i].append(curr_tile_pred["box_labels"][i])

        # arrays with box data for each batch
        joined_boxes = {"pred_boxes": [], "box_scores": [], "box_labels": []}
        for i in range(batch_size):
            # n in this case represents number of predicted boxes
            # stack boxes into form [n, 4] (vertical stack)
            joined_boxes["pred_boxes"].append(torch.vstack(boxes[i]))
            # stack scores and labels into form [n] (horizontal stack)
            joined_boxes["box_scores"].append(torch.hstack(scores[i]))
            joined_boxes["box_labels"].append(torch.hstack(labels[i]))

        return joined_boxes

    def join_labels_and_scores(self, batch_data: dict) -> dict[str, Tensor]:
        """
        Join scores and their corresponding label predictions from all tiles for each image.
        Label joining is done by rule where one anomalous tile in image results in whole image being anomalous.
        Scores are averaged over tiles.

        Args:
            batch_data: Dictionary containing all tile predictions of current batch.

        Returns:
            Dictionary with "pred_labels" and "pred_scores"
        """
        labels = torch.empty(batch_data[(0, 0)]["pred_labels"].shape, dtype=torch.bool)
        scores = torch.zeros(batch_data[(0, 0)]["pred_scores"].shape)

        for curr_tile_data in batch_data.values():
            curr_labels = curr_tile_data["pred_labels"]
            curr_scores = curr_tile_data["pred_scores"]

            labels = labels.logical_or(curr_labels)
            scores += curr_scores

        scores /= self.tiler.num_patches_h * self.tiler.num_patches_w

        joined = {"pred_labels": labels, "pred_scores": scores}

        return joined
