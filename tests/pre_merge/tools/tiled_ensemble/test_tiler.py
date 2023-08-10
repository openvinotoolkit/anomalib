"""EnsembleTiler tests"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch
from omegaconf import OmegaConf

from tools.tiled_ensemble.ensemble_tiler import EnsembleTiler

tiler_config = OmegaConf.create(
    {
        "ensemble": {
            "tiling": {
                "tile_size": 256,
                "stride": 256,
            }
        },
        "dataset": {"image_size": 512},
    }
)

tiler_config_overlap = OmegaConf.create(
    {
        "ensemble": {
            "tiling": {
                "tile_size": 256,
                "stride": 128,
            }
        },
        "dataset": {"image_size": 512},
    }
)


@pytest.mark.parametrize(
    "input_shape, config, expected_shape",
    [
        (torch.Size([5, 3, 512, 512]), tiler_config, torch.Size([2, 2, 5, 3, 256, 256])),
        (torch.Size([5, 3, 512, 512]), tiler_config_overlap, torch.Size([3, 3, 5, 3, 256, 256])),
        (torch.Size([5, 3, 500, 500]), tiler_config, torch.Size([2, 2, 5, 3, 256, 256])),
        (torch.Size([5, 3, 500, 500]), tiler_config_overlap, torch.Size([3, 3, 5, 3, 256, 256])),
    ],
)
def test_basic_tile_for_ensemble(input_shape, config, expected_shape):
    config = copy.deepcopy(config)
    config.dataset.image_size = input_shape[-1]
    tiler = EnsembleTiler(
        tile_size=config.ensemble.tiling.tile_size,
        stride=config.ensemble.tiling.stride,
        image_size=config.dataset.image_size,
    )

    images = torch.rand(size=input_shape)
    tiled = tiler.tile(images)

    assert tiled.shape == expected_shape


@pytest.mark.parametrize(
    "input_shape, config",
    [
        (torch.Size([5, 3, 512, 512]), tiler_config),
        (torch.Size([5, 3, 512, 512]), tiler_config_overlap),
        (torch.Size([5, 3, 500, 500]), tiler_config),
        (torch.Size([5, 3, 500, 500]), tiler_config_overlap),
    ],
)
def test_basic_tile_reconstruction(input_shape, config):
    config = copy.deepcopy(config)
    config.dataset.image_size = input_shape[-1]

    tiler = EnsembleTiler(
        tile_size=config.ensemble.tiling.tile_size,
        stride=config.ensemble.tiling.stride,
        image_size=config.dataset.image_size,
    )

    images = torch.rand(size=input_shape)
    tiled = tiler.tile(images.clone())
    untiled = tiler.untile(tiled)

    assert images.shape == untiled.shape
    assert images.equal(untiled)


@pytest.mark.parametrize(
    "input_shape, config",
    [
        (torch.Size([5, 3, 512, 512]), tiler_config),
        (torch.Size([5, 3, 500, 500]), tiler_config),
    ],
)
def test_untile_different_instance(input_shape, config):
    config = copy.deepcopy(config)
    config.dataset.image_size = input_shape[-1]
    tiler_1 = EnsembleTiler(
        tile_size=config.ensemble.tiling.tile_size,
        stride=config.ensemble.tiling.stride,
        image_size=config.dataset.image_size,
    )

    tiler_2 = EnsembleTiler(
        tile_size=config.ensemble.tiling.tile_size,
        stride=config.ensemble.tiling.stride,
        image_size=config.dataset.image_size,
    )

    images = torch.rand(size=input_shape)
    tiled = tiler_1.tile(images.clone())

    untiled = tiler_2.untile(tiled)

    # untiling should work even with different instance of tiler
    assert images.shape == untiled.shape
    assert images.equal(untiled)
