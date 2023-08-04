"""EnsembleTiler tests"""

import pytest
import torch
from omegaconf import OmegaConf

from anomalib.models.ensemble.ensemble_tiler import EnsembleTiler

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
    ],
)
def test_basic_tile_for_ensemble(input_shape, config, expected_shape):
    tiler = EnsembleTiler(config)

    images = torch.rand(size=input_shape)
    tiled = tiler.tile(images)

    assert tiled.shape == expected_shape


@pytest.mark.parametrize(
    "input_shape, config",
    [(torch.Size([5, 3, 512, 512]), tiler_config), (torch.Size([5, 3, 512, 512]), tiler_config_overlap)],
)
def test_basic_tile_reconstruction(input_shape, config):
    tiler = EnsembleTiler(config)

    images = torch.rand(size=input_shape)
    tiled = tiler.tile(images.clone())
    untiled = tiler.untile(tiled)

    assert images.shape == untiled.shape

    assert images.equal(untiled)
