"""Tiler used with ensemble of models."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import Any, Callable

from torch import Tensor

from anomalib.pre_processing.tiler import Tiler

from anomalib.data.base.datamodule import collate_fn


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


class TileTransform(object):
    """Tile the image and return tile

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size


def make_tile_colalte_fn(tiler: EnsembleTiler, index: (int, int)) -> Callable:
    def tile_collate_fn(batch: list) -> dict[str, Any]:
        coll_batch = collate_fn(batch)

        tiled_images = tiler.tile(coll_batch["image"])
        # insert channel (as mask has just one)
        tiled_masks = tiler.tile(coll_batch["mask"].unsqueeze(1))

        # remove batch
        batch["image"] = tiled_images[index]
        # remove channel
        batch["mask"] = tiled_masks[index].squeeze(1)

        return batch
