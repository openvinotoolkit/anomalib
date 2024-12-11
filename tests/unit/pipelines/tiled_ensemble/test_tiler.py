"""Tiling related tests for tiled ensemble."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch

from anomalib.data import AnomalibDataModule
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


class TestTiler:
    """EnsembleTiler tests."""

    @staticmethod
    @pytest.mark.parametrize(
        ("input_shape", "config", "expected_shape"),
        [
            (torch.Size([5, 3, 512, 512]), tiler_config, torch.Size([2, 2, 5, 3, 256, 256])),
            (torch.Size([5, 3, 512, 512]), tiler_config_overlap, torch.Size([3, 3, 5, 3, 256, 256])),
            (torch.Size([5, 3, 500, 500]), tiler_config, torch.Size([2, 2, 5, 3, 256, 256])),
            (torch.Size([5, 3, 500, 500]), tiler_config_overlap, torch.Size([3, 3, 5, 3, 256, 256])),
        ],
    )
    def test_basic_tile_for_ensemble(input_shape: torch.Size, config: dict, expected_shape: torch.Size) -> None:
        """Test basic tiling of data."""
        config = copy.deepcopy(config)
        config["data"]["init_args"]["image_size"] = input_shape[-1]
        tiler = get_ensemble_tiler(config["tiling"], config["data"])

        images = torch.rand(size=input_shape)
        tiled = tiler.tile(images)

        assert tiled.shape == expected_shape

    @staticmethod
    @pytest.mark.parametrize(
        ("input_shape", "config"),
        [
            (torch.Size([5, 3, 512, 512]), tiler_config),
            (torch.Size([5, 3, 512, 512]), tiler_config_overlap),
            (torch.Size([5, 3, 500, 500]), tiler_config),
            (torch.Size([5, 3, 500, 500]), tiler_config_overlap),
        ],
    )
    def test_basic_tile_reconstruction(input_shape: torch.Size, config: dict) -> None:
        """Test basic reconstruction of tiled data."""
        config = copy.deepcopy(config)
        config["data"]["init_args"]["image_size"] = input_shape[-1]

        tiler = get_ensemble_tiler(config["tiling"], config["data"])

        images = torch.rand(size=input_shape)
        tiled = tiler.tile(images.clone())
        untiled = tiler.untile(tiled)

        assert images.shape == untiled.shape
        assert images.equal(untiled)

    @staticmethod
    @pytest.mark.parametrize(
        ("input_shape", "config"),
        [
            (torch.Size([5, 3, 512, 512]), tiler_config),
            (torch.Size([5, 3, 500, 500]), tiler_config),
        ],
    )
    def test_untile_different_instance(input_shape: torch.Size, config: dict) -> None:
        """Test untiling with different Tiler instance."""
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


class TestTileCollater:
    """Test tile collater."""

    @staticmethod
    def test_collate_tile_shape(get_ensemble_config: dict, get_datamodule: AnomalibDataModule) -> None:
        """Test that collate function successfully tiles the image."""
        config = get_ensemble_config
        # datamodule with tile collater
        datamodule = get_datamodule

        tile_w, tile_h = config["tiling"]["tile_size"]

        batch = next(iter(datamodule.train_dataloader()))
        assert batch["image"].shape[1:] == (3, tile_w, tile_h)
        assert batch["mask"].shape[1:] == (tile_w, tile_h)
