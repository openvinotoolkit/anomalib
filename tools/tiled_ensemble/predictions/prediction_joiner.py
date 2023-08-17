"""Class used as mechanism to join/combine ensemble predictions."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC

from tools.tiled_ensemble.ensemble_tiler import EnsembleTiler
from tools.tiled_ensemble.predictions.prediction_data import EnsemblePredictions
from torch import Tensor


class EnsemblePredictionJoiner(ABC):
    """
    Class used for joining/combining the data predicted by each separate model of ensemble.

    Override methods to change how joining is done.

    Args:
        tiler (EnsembleTiler): Tiler used to transform tiles back to image level representation.

    """

    def __init__(self, tiler: EnsembleTiler) -> None:
        self.tiler = tiler

        self.ensemble_predictions: EnsemblePredictions = None
        self.num_batches = 0

    def setup(self, ensemble_predictions: EnsemblePredictions) -> None:
        """
        Prepare the joiner for given prediction data.

        Args:
            ensemble_predictions (EnsemblePredictions): Dictionary containing batched predictions for each tile.

        """
        assert ensemble_predictions.num_batches > 0, "There should be at least one batch for each tile prediction."
        assert (0, 0) in ensemble_predictions.get_batch_tiles(
            0
        ), "Tile prediction dictionary should always have at least one tile"

        self.ensemble_predictions = ensemble_predictions
        self.num_batches = self.ensemble_predictions.num_batches

    def join_tiles(self, batch_data: dict, tile_key: str) -> Tensor:
        """
        Join tiles back into one tensor and perform untiling with tiler.

        Args:
            batch_data (dict): Dictionary containing all tile predictions of current batch.
            tile_key (str): Key used in prediction dictionary for tiles that we want to join.

        Returns:
            Tensor: Tensor of tiles in original (stitched) shape.
        """
        raise NotImplementedError

    def join_boxes(self, batch_data: dict) -> dict[str, list[Tensor]]:
        """
        Join boxes data from all tiles. This includes pred_boxes, box_scores and box_labels.

        Args:
            batch_data (dict): Dictionary containing all tile predictions of current batch.

        Returns:
            dict[str, list[Tensor]]: Dictionary with joined boxes, box scores and box labels.
        """
        raise NotImplementedError

    def join_labels_and_scores(self, batch_data: dict) -> dict[str, Tensor]:
        """
        Join scores and their corresponding label predictions from all tiles for each image.

        Args:
            batch_data (dict): Dictionary containing all tile predictions of current batch.

        Returns:
            dict[str, Tensor]: Dictionary with "pred_labels" and "pred_scores"
        """
        raise NotImplementedError

    def join_tile_predictions(self, batch_index: int) -> dict[str, Tensor | list]:
        """
        Join predictions from ensemble into whole image level representation for batch at index batch_index.

        Args:
            batch_index (int): Index of current batch.

        Returns:
            dict[str, Tensor | list]: List of joined predictions for specified batch.
        """
        current_batch_data = self.ensemble_predictions.get_batch_tiles(batch_index)

        tiled_keys = ["image", "mask", "anomaly_maps", "pred_masks"]
        # take first tile as base prediction, keep items that are the same over all tiles:
        # image_path, label, mask_path
        joined_predictions = {
            "image_path": current_batch_data[(0, 0)]["image_path"],
            "label": current_batch_data[(0, 0)]["label"],
        }
        if "mask_path" in current_batch_data[(0, 0)].keys():
            joined_predictions["mask_path"] = current_batch_data[(0, 0)]["mask_path"]
        if "boxes" in current_batch_data[(0, 0)].keys():
            joined_predictions["boxes"] = current_batch_data[(0, 0)]["boxes"]

            # join all box data from all tiles
            joined_box_data = self.join_boxes(current_batch_data)
            joined_predictions["pred_boxes"] = joined_box_data["pred_boxes"]
            joined_predictions["box_scores"] = joined_box_data["box_scores"]
            joined_predictions["box_labels"] = joined_box_data["box_labels"]

        # join all tiled data
        for t_key in tiled_keys:
            if t_key in current_batch_data[(0, 0)].keys():
                joined_predictions[t_key] = self.join_tiles(current_batch_data, t_key)

        # label and score joining
        joined_scores_and_labels = self.join_labels_and_scores(current_batch_data)
        joined_predictions["pred_labels"] = joined_scores_and_labels["pred_labels"]
        joined_predictions["pred_scores"] = joined_scores_and_labels["pred_scores"]

        return joined_predictions
