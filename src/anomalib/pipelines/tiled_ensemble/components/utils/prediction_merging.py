"""Class used as mechanism to merge ensemble predictions from each tile into complete whole-image representation."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from anomalib.data.utils import masks_to_boxes

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

    def merge_boxes(self, batch_data: dict) -> dict:
        """Merge boxes data from all tiles. This includes pred_boxes, box_scores and box_labels.

        Joining is done by stacking boxes from all tiles.

        Args:
            batch_data (dict): Dictionary containing all tile predictions of current batch.

        Returns:
            dict: Dictionary with merged boxes, box scores and box labels.
        """
        # batch of tiles with index (0, 0) always exists, so we use it to get some basic information
        batch_size = len(batch_data[0, 0]["pred_boxes"])

        # create array of placeholder arrays, that will contain all boxes for each image
        boxes: list[list[Tensor]] = [[] for _ in range(batch_size)]
        scores: list[list[Tensor]] = [[] for _ in range(batch_size)]
        labels: list[list[Tensor]] = [[] for _ in range(batch_size)]

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
        merged_boxes: dict[str, list[Tensor]] = {"pred_boxes": [], "box_scores": [], "box_labels": []}
        for i in range(batch_size):
            # n in this case represents number of predicted boxes
            # stack boxes into form [n, 4] (vertical stack)
            merged_boxes["pred_boxes"].append(torch.vstack(boxes[i]))
            # stack scores and labels into form [n] (horizontal stack)
            merged_boxes["box_scores"].append(torch.hstack(scores[i]))
            merged_boxes["box_labels"].append(torch.hstack(labels[i]))

        return merged_boxes

    @staticmethod
    def generate_boxes(anomaly_maps: torch.Tensor, pred_masks: torch.Tensor) -> dict:
        """Merge box predictions from tiled data by recalculating them from merged anomaly maps.

        This produces pred_boxes, box_scores and box_labels.

        Args:
            anomaly_maps (torch.Tensor): Merged anomaly maps.
            pred_masks (torch.Tensor): Merged predicted anomaly masks.

        Returns:
            dict: Dictionary with merged boxes, box scores and box labels.
        """
        pred_boxes, box_scores = masks_to_boxes(
            pred_masks,
            anomaly_maps,
        )
        box_labels = [torch.ones(boxes.shape[0]) for boxes in pred_boxes]

        return {"pred_boxes": pred_boxes, "box_scores": box_scores, "box_labels": box_labels}

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

        if "pred_boxes" in current_batch_data[0, 0]:
            if "anomaly_maps" in merged_predictions:
                # if anomaly maps are predicted, generate new boxes from anomaly maps instead of merging
                merged_box_data = self.generate_boxes(
                    merged_predictions["anomaly_maps"],
                    merged_predictions["pred_masks"],
                )
            else:
                # otherwise merge boxes for each image
                merged_box_data = self.merge_boxes(current_batch_data)

            merged_predictions["pred_boxes"] = merged_box_data["pred_boxes"]
            merged_predictions["box_scores"] = merged_box_data["box_scores"]
            merged_predictions["box_labels"] = merged_box_data["box_labels"]

        return merged_predictions
