"""Tiler used with ensemble of models"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from torch import Tensor

from anomalib.pre_processing.tiler import Tiler


class EnsembleTiler(Tiler):
    def __init__(self, tile_size, stride, remove_border_count):
        super().__init__(tile_size=tile_size, stride=stride, remove_border_count=remove_border_count)

    def tile(self, images: Tensor) -> Tensor:
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
        # [num_h, num_w, batch, channel, tile_height, tile_width]
        _, _, _, channels, tile_size_h, tile_size_w = tiles.shape

        # rearrange the tiles in order [tile_count * batch, channels, tile_height, tile_width]
        # the required shape for untiling
        tiles = tiles.permute(2, 0, 1, 3, 4, 5)
        tiles = tiles.contiguous().view(-1, channels, tile_size_h, tile_size_w)

        untiled = super().untile(tiles)

        return untiled
