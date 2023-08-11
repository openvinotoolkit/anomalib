"""Tiler used with ensemble of models."""
from typing import Sequence

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from torch import Tensor

from anomalib.pre_processing.tiler import Tiler, compute_new_image_size


class EnsembleTiler(Tiler):
    """
    Tile Image into (non)overlapping Patches which are then used for ensemble training.

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
            image_size=self.image_size,
            tile_size=(self.tile_size_h, self.tile_size_w),
            stride=(self.stride_h, self.stride_w),
        )

        # get number of patches in both dimensions
        self.num_patches_h = int((self.resized_h - self.tile_size_h) / self.stride_h) + 1
        self.num_patches_w = int((self.resized_w - self.tile_size_w) / self.stride_w) + 1

    def tile(self, image: Tensor, use_random_tiling=False) -> Tensor:
        """
        Tiles an input image to either overlapping or non-overlapping patches.

        Args:
            image: Input images.
            use_random_tiling: Random tiling, which is part of original tiler but is unused here.

        Returns:
            Tiles generated from images. Returned shape: [num_h, num_w, batch, channel, tile_height, tile_width].
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

        return tiles

    def untile(self, tiles: Tensor) -> Tensor:
        """
        Reassemble the tiled tensor into image level representation.

        Args:
            tiles: Tiles in shape: [num_h, num_w, batch, channel, tile_height, tile_width].

        Returns:
            Image constructed from input tiles. Shape: [B, C, H, W].
        """
        # [num_h, num_w, batch, channel, tile_height, tile_width]
        _, _, batch, channels, tile_size_h, tile_size_w = tiles.shape

        # set tilers batch size as it might have been changed by previous tiling
        self.batch_size = batch

        # rearrange the tiles in order [tile_count * batch, channels, tile_height, tile_width]
        # the required shape for untiling
        tiles = tiles.permute(2, 0, 1, 3, 4, 5)
        tiles = tiles.contiguous().view(-1, channels, tile_size_h, tile_size_w)

        untiled = super().untile(tiles)

        return untiled
