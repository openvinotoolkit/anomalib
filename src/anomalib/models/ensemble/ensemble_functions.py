"""Functions used to train and use ensemble of models."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import Any
from itertools import product

import torch
from omegaconf import DictConfig, ListConfig
from torch import Tensor

from anomalib.data.base.datamodule import collate_fn
from anomalib.models.ensemble.ensemble_prediction_joiner import EnsemblePredictionJoiner
from anomalib.models.ensemble.ensemble_tiler import EnsembleTiler


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


def update_ensemble_input_size_config(config: DictConfig | ListConfig) -> DictConfig | ListConfig:
    """
    Update input size of model to match tile size.

    Args:
        config: Configurable parameters object

    Returns:
        Configurable parameters with updated values
    """
    tile_size = (config.dataset.tiling.tile_size,) * 2
    config.model.input_size = tile_size
    return config


class BasicPredictionJoiner(EnsemblePredictionJoiner):
    """
    Basic implementation of ensemble prediction joiner.
    Tiles are put together and untiled.
    Boxes are stacked per image.
    Labels are combined with OR operator, meaning one anomalous tile -> anomalous image

    """
    def join_tiles(self, batch_index: int, tile_key: str) -> Tensor:
        """
        Join tiles back into one tensor and perform untiling with tiler.

        Args:
            batch_index: Index of current batch.
            tile_key: Key used in prediction dictionary for tiles that we want to join

        Returns:
            Tensor of tiles in original (stitched) shape.
        """
        # tiles with index (0, 0) should always exist
        first_tiles = self.tile_predictions[(0, 0)][batch_index][tile_key]

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
        for tile_i, tile_j in product(range(self.tiler.num_patches_h), range(self.tiler.num_patches_w)):
            joined_masks[tile_i, tile_j, ...] = self.tile_predictions[(tile_i, tile_j)][batch_index][tile_key]

        if tile_key == "mask":
            # add channel as tiler needs it
            joined_masks = joined_masks.unsqueeze(3)

        # stitch tiles back into whole, output is [B, C, H, W]
        joined_output = self.tiler.untile(joined_masks)

        if tile_key == "mask":
            # remove previously added channels
            joined_output = joined_output.squeeze(1)

        return joined_output

    def join_boxes(self, batch_index: int) -> dict:
        """
        Join boxes data from all tiles. This includes pred_boxes, box_scores and box_labels.
        TODO: not correct joining, fix this!!

        Args:
            batch_index: Index of current batch.

        Returns:
            Dictionary with joined boxes, box scores and box labels.
        """
        joined_boxes = {"pred_boxes": [], "box_scores": [], "box_labels": []}

        # insert tile into joined tensor at right locations
        for tile_i, tile_j in product(range(self.tiler.num_patches_h), range(self.tiler.num_patches_w)):
            joined_boxes["pred_boxes"].extend(self.tile_predictions[(tile_i, tile_j)][batch_index]["pred_boxes"])
            joined_boxes["box_scores"].extend(self.tile_predictions[(tile_i, tile_j)][batch_index]["box_scores"])
            joined_boxes["box_labels"].extend(self.tile_predictions[(tile_i, tile_j)][batch_index]["box_labels"])

        return joined_boxes
