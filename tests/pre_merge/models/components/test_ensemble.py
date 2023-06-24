"""Model nsemble tests"""

import pytest

import torch

from omegaconf import DictConfig

from anomalib.models.ensemble import Ensemble

tiler_config = DictConfig(
    {
         "tiler": {
             "tile_size": 256,
             "stride": 256,
             "remove_border_count": 0
         }
    })

tiler_config_overlap = DictConfig(
    {
         "tiler": {
             "tile_size": 256,
             "stride": 128,
             "remove_border_count": 0
         }
    })

@pytest.mark.parametrize(
    "images_shape, config, expected_shape",
    [(torch.Size([5, 3, 512, 512]), tiler_config, torch.Size([2, 2, 5, 3, 256, 256])),
     (torch.Size([5, 3, 512, 512]), tiler_config_overlap, torch.Size([3, 3, 5, 3, 256, 256]))]
)
def test_basic_tile_for_ensemble(images_shape, config, expected_shape):
    ens = Ensemble(config)

    images = torch.rand(size=images_shape)
    tiled = ens.pre_process(images)

    assert tiled.shape == expected_shape


@pytest.mark.parametrize(
    "images_shape, config",
    [(torch.Size([5, 3, 512, 512]), tiler_config),
     (torch.Size([5, 3, 512, 512]), tiler_config_overlap)]
)
def test_basic_ensemble_reconstruction(images_shape, config):
    ens = Ensemble(config)

    images = torch.rand(size=images_shape)
    tiled = ens.pre_process(images.clone())
    untiled = ens.post_process(tiled)

    assert images.shape == untiled.shape

    assert images.equal(untiled)