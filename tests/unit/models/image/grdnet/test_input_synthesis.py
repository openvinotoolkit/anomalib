# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for GRD-Net input tiling and synthetic anomaly generation."""

import pytest
import torch

from anomalib.models.image.grdnet.anomaly_generator import (
    GRDNetAnomalyGenerator,
    _blend_anomaly,
    _sample_texture,
)
from anomalib.models.image.grdnet.tiling import rotate_image_and_roi, tile_image_and_roi


def test_tiling_uses_batch_major_row_major_order() -> None:
    """Canonical geometry should produce nine row-major tiles per image."""
    rows = torch.arange(256, dtype=torch.float32).view(1, 1, 256, 1)
    columns = torch.arange(256, dtype=torch.float32).view(1, 1, 1, 256)
    coordinates = rows * 256 + columns
    images = torch.cat((coordinates, coordinates, coordinates), dim=1).repeat(2, 1, 1, 1)
    images[1] += 100_000
    roi_masks = torch.ones((2, 1, 256, 256))

    image_tiles, roi_tiles = tile_image_and_roi(images, roi_masks)

    assert image_tiles.shape == (18, 3, 128, 128)
    assert roi_tiles.shape == (18, 1, 128, 128)
    assert image_tiles[:, 0, 0, 0].tolist() == [
        0,
        64,
        128,
        16_384,
        16_448,
        16_512,
        32_768,
        32_832,
        32_896,
        100_000,
        100_064,
        100_128,
        116_384,
        116_448,
        116_512,
        132_768,
        132_832,
        132_896,
    ]


def test_tiling_keeps_image_and_roi_aligned() -> None:
    """Concatenated tiling should preserve pixel alignment between inputs."""
    roi_masks = torch.zeros((1, 1, 256, 256))
    roi_masks[:, :, 40:200, 80:220] = 1
    images = roi_masks.repeat(1, 3, 1, 1)

    image_tiles, roi_tiles = tile_image_and_roi(images, roi_masks)

    assert torch.equal(image_tiles[:, :1], roi_tiles)


def test_rotation_uses_nearest_neighbor_for_roi() -> None:
    """ROI rotation should remain binary while sharing the image angle."""
    roi_masks = torch.zeros((1, 1, 32, 32))
    roi_masks[:, :, 8:24, 12:20] = 1
    images = roi_masks.repeat(1, 3, 1, 1)

    rotated_images, rotated_rois = rotate_image_and_roi(images, roi_masks, angles=torch.tensor([31.0]))

    assert set(rotated_rois.unique().tolist()) <= {0.0, 1.0}
    assert torch.all(rotated_images[:, :1][rotated_rois.bool()] > 0)


def test_probability_zero_returns_identity_and_empty_masks() -> None:
    """Skipped anomalies should not alter input tiles."""
    images = torch.rand((3, 3, 16, 16))
    generator = GRDNetAnomalyGenerator(probability=0.0)

    perturbed, textures, masks, beta = generator(images)

    assert torch.equal(perturbed, images)
    assert torch.count_nonzero(textures) == 0
    assert torch.count_nonzero(masks) == 0
    assert torch.all((beta >= 0.5) & (beta <= 1.0))


def test_blending_is_confined_to_mask() -> None:
    """The paper-canonical equation should blend only active mask pixels."""
    images = torch.full((1, 3, 2, 2), 0.25)
    textures = torch.full_like(images, 0.75)
    masks = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    beta = torch.tensor([[[[0.5]]]])

    perturbed = _blend_anomaly(images, textures, masks, beta)

    expected = torch.tensor([[[[0.5, 0.25], [0.25, 0.5]]]]).repeat(1, 3, 1, 1)
    assert torch.equal(perturbed, expected)


def test_random_texture_samples_independent_rgb_values() -> None:
    """Random textures should not repeat one scalar field across channels."""
    torch.manual_seed(7)
    texture = _sample_texture(torch.zeros((3, 16, 16)), "random")

    assert torch.all((texture >= 0.0) & (texture <= 1.0))
    assert not torch.equal(texture[0], texture[1])


def test_image_texture_is_a_nonzero_circular_shift() -> None:
    """Image textures should be in-distribution non-identity permutations."""
    torch.manual_seed(0)
    image = torch.arange(3 * 8 * 8, dtype=torch.float32).view(3, 8, 8)

    texture = _sample_texture(image, "image")

    assert not torch.equal(texture, image)
    assert torch.equal(texture.flatten().sort().values, image.flatten().sort().values)


def test_generated_mask_meets_area_and_output_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accepted Perlin masks should meet the area contract and produce finite output."""
    perlin = torch.zeros((8, 8))
    perlin[:5, :5] = 1
    monkeypatch.setattr(
        "anomalib.models.image.grdnet.anomaly_generator.generate_perlin_noise",
        lambda *_args, **_kwargs: perlin,
    )
    images = torch.full((1, 3, 8, 8), 0.5)
    generator = GRDNetAnomalyGenerator(probability=1.0, min_mask_area=25)

    perturbed, _, masks, beta = generator(images)

    assert masks.sum() == 25
    assert torch.all((beta >= 0.5) & (beta <= 1.0))
    assert torch.isfinite(perturbed).all()
    assert torch.all((perturbed >= 0.0) & (perturbed <= 1.0))


def test_generator_stops_after_configured_mask_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """An invalid Perlin sequence should fail with a bounded diagnostic."""
    attempts = 0

    def empty_perlin(*_args: object, **_kwargs: object) -> torch.Tensor:
        nonlocal attempts
        attempts += 1
        return torch.zeros((8, 8))

    monkeypatch.setattr(
        "anomalib.models.image.grdnet.anomaly_generator.generate_perlin_noise",
        empty_perlin,
    )
    generator = GRDNetAnomalyGenerator(probability=1.0, max_mask_attempts=8)

    with pytest.raises(RuntimeError, match="after 8 attempts"):
        generator(torch.zeros((1, 3, 8, 8)))
    assert attempts == 8
