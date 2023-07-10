"""Class used as mechanism to join/combine ensemble predictions."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from abc import ABC
from typing import List

from torch import Tensor

from anomalib.models.ensemble.ensemble_tiler import EnsembleTiler


class EnsemblePredictionJoiner(ABC):
    """
    Class used for joining/combining the data predicted by each separate model of ensemble.
    Override methods to change how joining is done.

    Args:
        tile_predictions: Dictionary containing batched predictions for each tile.
        tiler: Tiler used to transform tiles back to image level representation.
    """

    def __init__(self, tile_predictions: dict[(int, int), List], tiler: EnsembleTiler) -> None:
        assert (0, 0) in tile_predictions.keys(), "Tile prediction dictionary should always have at least one tile"
        assert tile_predictions[(0, 0)], "There should be at least one batch for each tile prediction."

        self.tile_predictions = tile_predictions
        self.tiler = tiler
        self.batch_count = len(tile_predictions[(0, 0)])

    def join_tiles(self, batch_index: int, tile_key: str) -> Tensor:
        """
        Join tiles back into one tensor and perform untiling with tiler.

        Args:
            batch_index: Index of current batch.
            tile_key: Key used in prediction dictionary for tiles that we want to join.

        Returns:
            Tensor of tiles in original (stitched) shape.
        """
        raise NotImplementedError

    def join_boxes(self, batch_index: int) -> dict[str, List[Tensor]]:
        """
        Join boxes data from all tiles. This includes pred_boxes, box_scores and box_labels.

        Args:
            batch_index: Index of current batch.

        Returns:
            Dictionary with joined boxes, box scores and box labels.
        """
        raise NotImplementedError

    def join_labels_and_scores(self, batch_index: int) -> dict[str, Tensor]:
        """
        Join scores and their corresponding label predictions from all tiles for each image.

        Args:
            batch_index: Index of current batch.

        Returns:
            Dictionary with "pred_labels" and "pred_scores"
        """
        raise NotImplementedError

    def join_tile_predictions(self) -> list[dict[str, Tensor | List | str]]:
        """
        Join predictions from ensemble into whole image level representation.

        Returns:
            List of joined predictions for each batch
        """
        joined_predictions = []

        tiled_keys = ["image", "mask", "anomaly_maps", "pred_masks"]
        # dict contains predictions for each tile location, and prediction is list of batches
        for batch_index in range(self.batch_count):
            # take first tile as base prediction, keep items that are the same over all tiles:
            # image_path, label, mask_path
            batch_predictions = {
                "image_path": self.tile_predictions[(0, 0)][batch_index]["image_path"],
                "label": self.tile_predictions[(0, 0)][batch_index]["label"],
                "mask_path": self.tile_predictions[(0, 0)][batch_index]["mask_path"]
            }

            # join all tiled data
            for t_key in tiled_keys:
                batch_predictions[t_key] = self.join_tiles(batch_index, t_key)

            # join all box data from all tiles
            joined_box_data = self.join_boxes(batch_index)
            batch_predictions["pred_boxes"] = joined_box_data["pred_boxes"]
            batch_predictions["box_scores"] = joined_box_data["box_scores"]
            batch_predictions["box_labels"] = joined_box_data["box_labels"]

            # label and score joining
            joined_scores_and_labels = self.join_labels_and_scores(batch_index)
            batch_predictions["pred_labels"] = joined_scores_and_labels["pred_labels"]
            batch_predictions["pred_scores"] = joined_scores_and_labels["pred_scores"]

            joined_predictions.append(batch_predictions)

        return joined_predictions
