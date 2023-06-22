"""Ensemble of models for tiled images"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum

import torch
from omegaconf import DictConfig, ListConfig

from torch import Tensor

from anomalib.pre_processing.tiler import Tiler


class TileGroupingMode(str, Enum):
    """
        Type of mode when processing tiles
        either in groups or each separately
    """
    GROUPED = "grouped"
    SEPARATE = "separate"


class Ensemble:
    """Prepare and train ensemble of models on tiled input images.

    """
    def __init__(
        self,
        config: DictConfig | ListConfig
    ) -> None:
        self.tiler_config = config.tiler
        self.tiler = Tiler(tile_size=self.tiler_config.tile_size,
                           stride=self.tiler_config.stride,
                           remove_border_count=self.tiler_config.remove_border_count)
        self.tiles: Tensor

    def pre_process(self, images: Tensor):
        batch, channels, _, _ = images.shape

        # tiles are returned in order [tile_count * batch, channels, tile_height, tile_width]
        combined_tiles = self.tiler.tile(images)

        # rearrange to [num_h, num_w, batch, channel, tile_height, tile_width]
        tiles = combined_tiles.contiguous().view(batch,
                                                 self.tiler.num_patches_h,
                                                 self.tiler.num_patches_w,
                                                 channels,
                                                 self.tiler.tile_size_h,
                                                 self.tiler.tile_size_w)
        tiles = tiles.permute(1, 2, 0, 3, 4, 5)

        return tiles

    def post_process(self, tiles: Tensor) -> Tensor:
        # [num_h, num_w, batch, channel, tile_height, tile_width]
        _, _, _, channels, tile_size_h, tile_size_w = tiles.shape

        # rearrange the tiles in order [tile_count * batch, channels, tile_height, tile_width]
        # the required shape for untiling
        tiles = tiles.permute(2, 0, 1, 3, 4, 5)
        tiles = tiles.contiguous().view(-1, channels, tile_size_h, tile_size_w)

        untiled = self.tiler.untile(tiles)
        return untiled

    def train(self):
        pass


if __name__ == "__main__":
    config = {
        "tiler": {
            "tile_size": 256,
            "stride": 128,
            "remove_border_count": 0
        }
    }
    config = DictConfig(config)
    ens = Ensemble(config)

    images = torch.rand(size=(5, 3, 512, 512))
    tiled = ens.pre_process(images.clone())
    print(tiled.shape)
    untiled = ens.post_process(tiled)
    print(untiled.shape)

    print(images.equal(untiled))
    print(images[0, 0, 0, 125:135])
    print(untiled[0, 0, 0, 125:135])
