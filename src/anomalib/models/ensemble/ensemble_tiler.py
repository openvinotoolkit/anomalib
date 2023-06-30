"""Tiler used with ensemble of models."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import Any, Sequence

from torch import Tensor

from anomalib.pre_processing.tiler import Tiler, compute_new_image_size


class EnsembleTiler(Tiler):
    """
    Tile Image into (non)overlapping Patches which are then used for ensemble training.

    Args:
        tile_size: Tile dimension for each patch.
        stride: Stride length between patches.
        image_size: Size of images that will be tiled.
        remove_border_count: Number of border pixels to be removed from tile before untiling.
    """
    def __init__(
            self,
            tile_size: int | Sequence,
            stride: int | Sequence,
            image_size: int | Sequence,
            remove_border_count: int = 0,
    ):
        super().__init__(tile_size=tile_size, stride=stride, remove_border_count=remove_border_count)

        # calculate final image size
        image_size = self._validate_size_type(image_size)
        self.resized_h, self.resized_w = compute_new_image_size(
            image_size=image_size,
            tile_size=(self.tile_size_h, self.tile_size_w),
            stride=(self.stride_h, self.stride_w),
        )

        # get number of patches in both dimensions
        self.num_patches_h = int((self.resized_h - self.tile_size_h) / self.stride_h) + 1
        self.num_patches_w = int((self.resized_w - self.tile_size_w) / self.stride_w) + 1

    def tile(self, images: Tensor) -> Tensor:
        """
        Tiles an input image to either overlapping or non-overlapping patches.

        Args:
            images: Input images.

        Returns:
            Tiles generated from images. Returned shape: [num_h, num_w, batch, channel, tile_height, tile_width].
        """
        # tiles are returned in order [tile_count * batch, channels, tile_height, tile_width]
        combined_tiles = super().tile(images)

        # rearrange to [num_h, num_w, batch, channel, tile_height, tile_width]
        tiles = combined_tiles.contiguous().view(self.batch_size,
                                                 self.num_patches_h,
                                                 self.num_patches_w,
                                                 self.num_channels,
                                                 self.tile_size_h,
                                                 self.tile_size_w)
        tiles = tiles.permute(1, 2, 0, 3, 4, 5)

        return tiles

    def untile(self, tiles: Tensor) -> Tensor:
        """

        Args:
            tiles: Tiles in shape: [num_h, num_w, batch, channel, tile_height, tile_width].

        Returns:
            Image constructed from input tiles. Shape: [B, C, H, W].
        """
        # [num_h, num_w, batch, channel, tile_height, tile_width]
        _, _, _, channels, tile_size_h, tile_size_w = tiles.shape

        # rearrange the tiles in order [tile_count * batch, channels, tile_height, tile_width]
        # the required shape for untiling
        tiles = tiles.permute(2, 0, 1, 3, 4, 5)
        tiles = tiles.contiguous().view(-1, channels, tile_size_h, tile_size_w)

        untiled = super().untile(tiles)

        return untiled
