"""Image Tiling Tests."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from omegaconf import ListConfig

from anomalib.data.utils.tiler import ImageUpscaleMode, StrideSizeError, Tiler

tile_data = [
    ([3, 1024, 1024], 512, 512, torch.Size([4, 3, 512, 512]), False),
    ([1, 3, 1024, 1024], 512, 512, torch.Size([4, 3, 512, 512]), False),
    ([3, 1024, 1024], 512, 512, torch.Size([4, 3, 512, 512]), True),
    ([1, 3, 1024, 1024], 512, 512, torch.Size([4, 3, 512, 512]), True),
]

untile_data = [
    (512, 256, torch.Size([4, 3, 512, 512])),
    (512, 512, torch.Size([4, 3, 512, 512])),
]

overlapping_data = [
    (
        torch.Size([1, 3, 1024, 1024]),
        512,
        256,
        "padding",
    ),
    (
        torch.Size([1, 3, 1024, 1024]),
        512,
        256,
        "interpolation",
    ),
]


@pytest.mark.parametrize(
    ("tile_size", "stride"),
    [(512, 256), ([512, 512], [256, 256]), (ListConfig([512, 512]), 256)],
)
def test_size_types_should_be_int_tuple_or_list_config(tile_size: tuple, stride: tuple) -> None:
    """Size type could only be integer, tuple or ListConfig type."""
    tiler = Tiler(tile_size=tile_size, stride=stride)
    assert isinstance(tiler.tile_size_h, int)
    assert isinstance(tiler.stride_w, int)


@pytest.mark.parametrize(("image_size", "tile_size", "stride", "shape", "use_random_tiling"), tile_data)
def test_tiler_handles_single_image_without_batch_dimension(
    image_size: tuple,
    tile_size: int,
    stride: int,
    shape: torch.Size,
    use_random_tiling: bool,
) -> None:
    """Tiler should add batch dimension if image is 3D (CxHxW)."""
    tiler = Tiler(tile_size=tile_size, stride=stride)
    image = torch.rand(image_size)
    patches = tiler.tile(image, use_random_tiling=use_random_tiling)
    assert patches.shape == shape


def test_stride_size_cannot_be_larger_than_tile_size() -> None:
    """Larger stride size than tile size is not desired, and causes issues."""
    kernel_size = (128, 128)
    stride = 256
    with pytest.raises(StrideSizeError):
        Tiler(tile_size=kernel_size, stride=stride)


def test_tile_size_cannot_be_larger_than_image_size() -> None:
    """Larger tile size than image size is not desired, and causes issues."""
    tiler = Tiler(tile_size=1024, stride=512)
    image = torch.rand(1, 3, 512, 512)
    with pytest.raises(ValueError):  # noqa: PT011
        tiler.tile(image)


@pytest.mark.parametrize(("kernel_size", "stride", "image_size"), untile_data)
def test_untile_non_overlapping_patches(kernel_size: int, stride: int, image_size: torch.Size) -> None:
    """Non-Overlapping Tiling/Untiling should return the same image size."""
    tiler = Tiler(tile_size=kernel_size, stride=stride)
    image = torch.rand(image_size)
    tiles = tiler.tile(image)

    untiled_image = tiler.untile(tiles)
    assert untiled_image.shape == torch.Size(image_size)


@pytest.mark.parametrize("mode", ["pad", "padded", "interpolate", "interplation"])
def test_upscale_downscale_mode(mode: str) -> None:
    """Test incorrect upscale mode."""
    with pytest.raises(ValueError):  # noqa: PT011
        Tiler(tile_size=(512, 512), stride=(256, 256), mode=mode)


@pytest.mark.parametrize(("image_size", "kernel_size", "stride", "mode"), overlapping_data)
@pytest.mark.parametrize("remove_border_count", [0, 5])
def test_untile_overlapping_patches(
    image_size: tuple,
    kernel_size: int,
    stride: int,
    remove_border_count: int,
    mode: ImageUpscaleMode,
) -> None:
    """Overlapping Tiling/Untiling should return the same image size."""
    tiler = Tiler(
        tile_size=kernel_size,
        stride=stride,
        remove_border_count=remove_border_count,
        mode=mode,
    )

    image = torch.rand(image_size)
    tiles = tiler.tile(image)
    reconstructed_image = tiler.untile(tiles)
    image = image[
        :,
        :,
        remove_border_count:-remove_border_count,
        remove_border_count:-remove_border_count,
    ]
    reconstructed_image = reconstructed_image[
        :,
        :,
        remove_border_count:-remove_border_count,
        remove_border_count:-remove_border_count,
    ]
    assert torch.equal(image, reconstructed_image)


@pytest.mark.parametrize("image_size", [(1, 3, 512, 512)])
@pytest.mark.parametrize("tile_size", [(256, 256), (200, 200), (211, 213), (312, 333), (511, 511)])
@pytest.mark.parametrize("stride", [(64, 64), (111, 111), (128, 111), (128, 128)])
@pytest.mark.parametrize("mode", [ImageUpscaleMode.PADDING, ImageUpscaleMode.INTERPOLATION])
def test_divisible_tile_size_and_stride(
    image_size: tuple,
    tile_size: tuple,
    stride: tuple,
    mode: ImageUpscaleMode,
) -> None:
    """Test non-divisible image size.

    When the image is not divisible by tile size and stride, Tiler should up
    samples the image before tiling, and downscales before untiling.
    """
    tiler = Tiler(tile_size, stride, mode=mode)
    image = torch.rand(image_size)
    tiles = tiler.tile(image)
    reconstructed_image = tiler.untile(tiles)
    assert image.shape == reconstructed_image.shape

    if mode == "padding":
        assert torch.allclose(image, reconstructed_image)
