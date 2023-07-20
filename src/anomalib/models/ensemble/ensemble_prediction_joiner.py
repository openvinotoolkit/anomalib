"""Class used as mechanism to join/combine ensemble predictions."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import List

from torch import Tensor

from anomalib.models.ensemble.ensemble_prediction_data import EnsemblePredictions
from anomalib.models.ensemble.ensemble_tiler import EnsembleTiler


class EnsemblePredictionJoiner(ABC):
    """
    Class used for joining/combining the data predicted by each separate model of ensemble.
    Override methods to change how joining is done.

    Args:
        ensemble_predictions: Dictionary containing batched predictions for each tile.
        tiler: Tiler used to transform tiles back to image level representation.
    """

    def __init__(self, ensemble_predictions: EnsemblePredictions, tiler: EnsembleTiler) -> None:
        assert ensemble_predictions.num_batches > 0, "There should be at least one batch for each tile prediction."
        assert (0, 0) in ensemble_predictions.get_batch_tiles(
            0
        ), "Tile prediction dictionary should always have at least one tile"

        self.ensemble_predictions = ensemble_predictions
        self.tiler = tiler

    def join_tiles(self, batch_data: dict, tile_key: str) -> Tensor:
        """
        Join tiles back into one tensor and perform untiling with tiler.

        Args:
            batch_data: Dictionary containing all tile predictions of current batch.
            tile_key: Key used in prediction dictionary for tiles that we want to join.

        Returns:
            Tensor of tiles in original (stitched) shape.
        """
        raise NotImplementedError

    def join_boxes(self, batch_data: dict) -> dict[str, List[Tensor]]:
        """
        Join boxes data from all tiles. This includes pred_boxes, box_scores and box_labels.

        Args:
            batch_data: Dictionary containing all tile predictions of current batch.

        Returns:
            Dictionary with joined boxes, box scores and box labels.
        """
        raise NotImplementedError

    def join_labels_and_scores(self, batch_data: dict) -> dict[str, Tensor]:
        """
        Join scores and their corresponding label predictions from all tiles for each image.

        Args:
            batch_data: Dictionary containing all tile predictions of current batch.

        Returns:
            Dictionary with "pred_labels" and "pred_scores"
        """
        raise NotImplementedError

    def join_tile_predictions(self, batch_index: int) -> dict[str, Tensor | List | str]:
        """
        Join predictions from ensemble into whole image level representation.

        Args:
            batch_index: Index of current batch.

        Returns:
            List of joined predictions for each batch
        """
        current_batch_data = self.ensemble_predictions.get_batch_tiles(batch_index)

        tiled_keys = ["image", "mask", "anomaly_maps", "pred_masks"]
        # take first tile as base prediction, keep items that are the same over all tiles:
        # image_path, label, mask_path
        joined_predictions = {
            "image_path": current_batch_data[(0, 0)]["image_path"],
            "label": current_batch_data[(0, 0)]["label"],
            "mask_path": current_batch_data[(0, 0)]["mask_path"],
        }

        # join all tiled data
        for t_key in tiled_keys:
            joined_predictions[t_key] = self.join_tiles(current_batch_data, t_key)

        # join all box data from all tiles
        joined_box_data = self.join_boxes(current_batch_data)
        joined_predictions["pred_boxes"] = joined_box_data["pred_boxes"]
        joined_predictions["box_scores"] = joined_box_data["box_scores"]
        joined_predictions["box_labels"] = joined_box_data["box_labels"]

        # label and score joining
        joined_scores_and_labels = self.join_labels_and_scores(current_batch_data)
        joined_predictions["pred_labels"] = joined_scores_and_labels["pred_labels"]
        joined_predictions["pred_scores"] = joined_scores_and_labels["pred_scores"]

        return joined_predictions
