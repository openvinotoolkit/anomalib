"""Functions used to train and use ensemble of models."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import Any
from itertools import product

import torch
from omegaconf import DictConfig, ListConfig
from torch import Tensor

from anomalib.data.base.datamodule import collate_fn
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


def join_tile_predictions(tile_predictions: dict, tiler: EnsembleTiler) -> list[Tensor]:
    assert (0, 0) in tile_predictions.keys(), "Tile prediction dictionary should always have at least one tile"

    batch_count = len(tile_predictions[(0, 0)])

    batch_masks = []

    # dict contains predictions for each tile location, and prediction is list of batches
    for batch_index in range(batch_count):
        # get batch, channels and device from first tile (0, 0) that should always exist
        batch_size, num_channels, _, _ = tile_predictions[(0, 0)][batch_index]["pred_masks"].shape
        device = tile_predictions[(0, 0)][batch_index]["pred_masks"].device

        joined_masks = torch.zeros(
            (tiler.num_patches_h, tiler.num_patches_w, batch_size, num_channels, tiler.tile_size_h, tiler.tile_size_w),
            device=device,
        )

        for tile_i, tile_j in product(range(tiler.num_patches_h), range(tiler.num_patches_w)):
            joined_masks[tile_i, tile_j, ...] = tile_predictions[(tile_i, tile_j)][batch_index]["pred_masks"]

        untiled_masks = tiler.untile(joined_masks)
        batch_masks.append(untiled_masks)

    return batch_masks
