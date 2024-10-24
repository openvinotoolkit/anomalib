"""Class used as mechanism to merge ensemble predictions from each tile into complete whole-image representation."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from .ensemble_tiling import EnsembleTiler
from .prediction_data import EnsemblePredictions


class PredictionMergingMechanism:
    """Class used for merging the data predicted by each separate model of tiled ensemble.

    Tiles are stacked in one tensor and untiled using Ensemble Tiler.
    Boxes from tiles are either stacked or generated anew from anomaly map.
    Labels are combined with OR operator, meaning one anomalous tile -> anomalous image.
    Scores are averaged across all tiles.

    Args:
        ensemble_predictions (EnsemblePredictions): Object containing predictions on tile level.
        tiler (EnsembleTiler): Tiler used to transform tiles back to image level representation.

    Example:
        >>> from anomalib.pipelines.tiled_ensemble.components.utils.ensemble_tiling import EnsembleTiler
        >>> from anomalib.pipelines.tiled_ensemble.components.utils.prediction_data import EnsemblePredictions
        >>>
        >>> tiler = EnsembleTiler(tile_size=256, stride=128, image_size=512)
        >>> data = EnsemblePredictions()
        >>> merger = PredictionMergingMechanism(data, tiler)
        >>>
        >>> # we can then start merging procedure for each batch
        >>> merger.merge_tile_predictions(0)
    """

    def __init__(self, ensemble_predictions: EnsemblePredictions, tiler: EnsembleTiler) -> None:
        assert ensemble_predictions.num_batches > 0, "There should be at least one batch for each tile prediction."
        assert (0, 0) in ensemble_predictions.get_batch_tiles(
            0,
        ), "Tile prediction dictionary should always have at least one tile"

        self.ensemble_predictions = ensemble_predictions
        self.num_batches = self.ensemble_predictions.num_batches

        self.tiler = tiler

    def merge_tiles(self, batch_data: dict, tile_key: str) -> Tensor:
        """Merge tiles back into one tensor and perform untiling with tiler.

        Args:
            batch_data (dict): Dictionary containing all tile predictions of current batch.
            tile_key (str): Key used in prediction dictionary for tiles that we want to merge.

        Returns:
            Tensor: Tensor of tiles in original (stitched) shape.
        """
        # batch of tiles with index (0, 0) always exists, so we use it to get some basic information
        first_tiles = batch_data[0, 0][tile_key]
        batch_size = first_tiles.shape[0]
        device = first_tiles.device

        if tile_key == "mask":
            # in case of ground truth masks, we don't have channels
            merged_size = [
                self.tiler.num_patches_h,
                self.tiler.num_patches_w,
                batch_size,
                self.tiler.tile_size_h,
                self.tiler.tile_size_w,
            ]
        else:
            # all tiles beside masks also have channels
            num_channels = first_tiles.shape[1]
            merged_size = [
                self.tiler.num_patches_h,
                self.tiler.num_patches_w,
                batch_size,
                int(num_channels),
                self.tiler.tile_size_h,
                self.tiler.tile_size_w,
            ]

        # create new empty tensor for merged tiles
        merged_masks = torch.zeros(size=merged_size, device=device)

        # insert tile into merged tensor at right locations
        for (tile_i, tile_j), tile_data in batch_data.items():
            merged_masks[tile_i, tile_j, ...] = tile_data[tile_key]

        if tile_key == "mask":
            # add channel as tiler needs it
            merged_masks = merged_masks.unsqueeze(3)

        # stitch tiles back into whole, output is [B, C, H, W]
        merged_output = self.tiler.untile(merged_masks)

        if tile_key == "mask":
            # remove previously added channels
            merged_output = merged_output.squeeze(1)

        return merged_output

    def merge_labels_and_scores(self, batch_data: dict) -> dict[str, Tensor]:
        """Join scores and their corresponding label predictions from all tiles for each image.

        Label merging is done by rule where one anomalous tile in image results in whole image being anomalous.
        Scores are averaged over tiles.

        Args:
            batch_data (dict): Dictionary containing all tile predictions of current batch.

        Returns:
            dict[str, Tensor]: Dictionary with "pred_labels" and "pred_scores"
        """
        # create accumulator with same shape as original
        labels = torch.zeros(batch_data[0, 0]["pred_labels"].shape, dtype=torch.bool)
        scores = torch.zeros(batch_data[0, 0]["pred_scores"].shape)

        for curr_tile_data in batch_data.values():
            curr_labels = curr_tile_data["pred_labels"]
            curr_scores = curr_tile_data["pred_scores"]

            labels = labels.logical_or(curr_labels)
            scores += curr_scores

        scores /= self.tiler.num_tiles

        return {"pred_labels": labels, "pred_scores": scores}

    def merge_tile_predictions(self, batch_index: int) -> dict[str, Tensor | list]:
        """Join predictions from ensemble into whole image level representation for batch at index batch_index.

        Args:
            batch_index (int): Index of current batch.

        Returns:
            dict[str, Tensor | list]: List of merged predictions for specified batch.
        """
        current_batch_data = self.ensemble_predictions.get_batch_tiles(batch_index)

        # take first tile as base prediction, keep items that are the same over all tiles:
        # image_path, label, mask_path
        merged_predictions = {
            "image_path": current_batch_data[0, 0]["image_path"],
            "label": current_batch_data[0, 0]["label"],
        }
        if "mask_path" in current_batch_data[0, 0]:
            merged_predictions["mask_path"] = current_batch_data[0, 0]["mask_path"]
        if "boxes" in current_batch_data[0, 0]:
            merged_predictions["boxes"] = current_batch_data[0, 0]["boxes"]

        tiled_data = ["image", "mask"]
        if "anomaly_maps" in current_batch_data[0, 0]:
            tiled_data += ["anomaly_maps", "pred_masks"]

        # merge all tiled data
        for t_key in tiled_data:
            if t_key in current_batch_data[0, 0]:
                merged_predictions[t_key] = self.merge_tiles(current_batch_data, t_key)

        # label and score merging
        merged_scores_and_labels = self.merge_labels_and_scores(current_batch_data)
        merged_predictions["pred_labels"] = merged_scores_and_labels["pred_labels"]
        merged_predictions["pred_scores"] = merged_scores_and_labels["pred_scores"]

        return merged_predictions
