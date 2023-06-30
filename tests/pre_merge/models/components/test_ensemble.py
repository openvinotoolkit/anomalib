"""Model nsemble tests"""

import pytest

import torch

from anomalib.models.ensemble.ensemble_tiler import EnsembleTiler

tiler_config = {
    "tile_size": 256,
    "stride": 256,
    "remove_border_count": 0
}

tiler_config_overlap = {
    "tile_size": 256,
    "stride": 128,
    "remove_border_count": 0
}


@pytest.mark.parametrize(
    "images_shape, config, expected_shape",
    [(torch.Size([5, 3, 512, 512]), tiler_config, torch.Size([2, 2, 5, 3, 256, 256])),
     (torch.Size([5, 3, 512, 512]), tiler_config_overlap, torch.Size([3, 3, 5, 3, 256, 256]))]
)
def test_basic_tile_for_ensemble(images_shape, config, expected_shape):
    tiler = EnsembleTiler(image_size=images_shape[2:], **config)

    images = torch.rand(size=images_shape)
    tiled = tiler.tile(images)

    assert tiled.shape == expected_shape


@pytest.mark.parametrize(
    "images_shape, config",
    [(torch.Size([5, 3, 512, 512]), tiler_config),
     (torch.Size([5, 3, 512, 512]), tiler_config_overlap)]
)
def test_basic_tile_reconstruction(images_shape, config):
    tiler = EnsembleTiler(image_size=images_shape[2:], **config)

    images = torch.rand(size=images_shape)
    tiled = tiler.tile(images.clone())
    untiled = tiler.untile(tiled)

    assert images.shape == untiled.shape

    assert images.equal(untiled)
