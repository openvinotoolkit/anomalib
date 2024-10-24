"""Tiler used with ensemble of models."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any

from torch import Tensor

from anomalib.data.base.datamodule import collate_fn
from anomalib.data.utils.tiler import Tiler, compute_new_image_size


class EnsembleTiler(Tiler):
    """Tile Image into (non)overlapping Patches which are then used for ensemble training.

    Args:
        tile_size (int | Sequence): Tile dimension for each patch.
        stride (int | Sequence): Stride length between patches.
        image_size (int | Sequence): Size of input image that will be tiled.

    Examples:
        >>> import torch
        >>> tiler = EnsembleTiler(tile_size=256, stride=128, image_size=512)
        >>>
        >>> # random images, shape:  [B, C, H, W]
        >>> images = torch.rand(32, 5, 512, 512)
        >>> # once tiled, the shape is [tile_count_H, tile_count_W, B, C, tile_H, tile_W]
        >>> tiled = tiler.tile(images)
        >>> tiled.shape
        torch.Size([3, 3, 32, 5, 256, 256])

        >>> # assemble the tiles back together
        >>> untiled = tiler.untile(tiled)
        >>> untiled.shape
        torch.Size([32, 5, 512, 512])
    """

    def __init__(self, tile_size: int | Sequence, stride: int | Sequence, image_size: int | Sequence) -> None:
        super().__init__(
            tile_size=tile_size,
            stride=stride,
        )

        # calculate final image size
        self.image_size = self.validate_size_type(image_size)
        self.input_h, self.input_w = self.image_size
        self.resized_h, self.resized_w = compute_new_image_size(
            image_size=(self.input_h, self.input_w),
            tile_size=(self.tile_size_h, self.tile_size_w),
            stride=(self.stride_h, self.stride_w),
        )

        # get number of patches in both dimensions
        self.num_patches_h = int((self.resized_h - self.tile_size_h) / self.stride_h) + 1
        self.num_patches_w = int((self.resized_w - self.tile_size_w) / self.stride_w) + 1
        self.num_tiles = self.num_patches_h * self.num_patches_w

    def tile(self, image: Tensor, use_random_tiling: bool = False) -> Tensor:
        """Tiles an input image to either overlapping or non-overlapping patches.

        Args:
            image (Tensor): Input images.
            use_random_tiling (bool): Random tiling, which is part of original tiler but is unused here.

        Returns:
            Tensor: Tiles generated from images.
                Returned shape: [num_h, num_w, batch, channel, tile_height, tile_width].
        """
        # tiles are returned in order [tile_count * batch, channels, tile_height, tile_width]
        combined_tiles = super().tile(image, use_random_tiling)

        # rearrange to [num_h, num_w, batch, channel, tile_height, tile_width]
        tiles = combined_tiles.contiguous().view(
            self.batch_size,
            self.num_patches_h,
            self.num_patches_w,
            self.num_channels,
            self.tile_size_h,
            self.tile_size_w,
        )
        tiles = tiles.permute(1, 2, 0, 3, 4, 5)

        return tiles  # noqa: RET504

    def untile(self, tiles: Tensor) -> Tensor:
        """Reassemble the tiled tensor into image level representation.

        Args:
            tiles (Tensor): Tiles in shape: [num_h, num_w, batch, channel, tile_height, tile_width].

        Returns:
            Tensor: Image constructed from input tiles. Shape: [B, C, H, W].
        """
        # tiles have shape [num_h, num_w, batch, channel, tile_height, tile_width]
        _, _, batch, channels, tile_size_h, tile_size_w = tiles.shape

        # set tilers batch size as it might have been changed by previous tiling
        self.batch_size = batch

        # rearrange the tiles in order [tile_count * batch, channels, tile_height, tile_width]
        # the required shape for untiling
        tiles = tiles.permute(2, 0, 1, 3, 4, 5)
        tiles = tiles.contiguous().view(-1, channels, tile_size_h, tile_size_w)

        untiled = super().untile(tiles)

        return untiled  # noqa: RET504


class TileCollater:
    """Class serving as collate function to perform tiling on batch of images from Dataloader.

    Args:
        tiler (EnsembleTiler): Tiler used to split the images to tiles.
        tile_index (tuple[int, int]): Index of tile we want to return.
    """

    def __init__(self, tiler: EnsembleTiler, tile_index: tuple[int, int]) -> None:
        self.tiler = tiler
        self.tile_index = tile_index

    def __call__(self, batch: list) -> dict[str, Any]:
        """Collate batch and tile images + masks from batch.

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

        if "mask" in coll_batch:
            # insert channel (as mask has just one)
            tiled_masks = self.tiler.tile(coll_batch["mask"].unsqueeze(1))

            # return only tiled at given index, squeeze to remove previously added channel
            coll_batch["mask"] = tiled_masks[self.tile_index].squeeze(1)

        return coll_batch
