"""EnsembleTiler tests"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch
from omegaconf import OmegaConf
from anomalib.pipelines.tiled_ensemble.components.utils.helper_functions import get_ensemble_tiler

tiler_config = {
        "tiling": {
            "tile_size": 256,
            "stride": 256,
        },
        "data": {"init_args": {"image_size": 512}},
    }

tiler_config_overlap = {
        "tiling": {
            "tile_size": 256,
            "stride": 128,
        },
        "data": {"init_args": {"image_size": 512}},
    }


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
    config["data"]["init_args"]["image_size"] = input_shape[-1]
    tiler = get_ensemble_tiler(config["tiling"], config["data"])

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
    config["data"]["init_args"]["image_size"] = input_shape[-1]

    tiler = get_ensemble_tiler(config["tiling"], config["data"])

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
    config["data"]["init_args"]["image_size"] = input_shape[-1]
    tiler_1 = get_ensemble_tiler(config["tiling"], config["data"])

    tiler_2 = get_ensemble_tiler(config["tiling"], config["data"])

    images = torch.rand(size=input_shape)
    tiled = tiler_1.tile(images.clone())

    untiled = tiler_2.untile(tiled)

    # untiling should work even with different instance of tiler
    assert images.shape == untiled.shape
    assert images.equal(untiled)
