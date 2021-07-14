import pytest
import torch
from omegaconf import ListConfig

from anomalib.datasets.utils import StrideSizeError, Tiler

# def test_unfold_handles_num_non_overlapping_patches():
#     batch_size, num_channels, height, width = 2, 3, 1024, 1024
#     image = torch.rand(batch_size, num_channels, height, width)
#     tiler = Tiler(batch_size, (height, width), kernel_size=512, stride=512)
#
#     patches = tiler.tile_image(image)
#
#     print(patches.shape)

tile_data = [
    ([3, 1024, 1024], 512, 512, torch.Size([4, 3, 512, 512])),
    ([1, 3, 1024, 1024], 512, 512, torch.Size([4, 3, 512, 512])),
]

untile_data = [
    ([3, 1024, 1024], 512, 256, torch.Size([4, 3, 512, 512])),
    ([1, 3, 1024, 1024], 512, 512, torch.Size([4, 3, 512, 512])),
]

overlapping_data = [
    (torch.Size([1, 3, 1024, 1024]), 512, 256, torch.Size([16, 3, 512, 512]), "padding"),
    (torch.Size([1, 3, 1024, 1024]), 512, 256, torch.Size([16, 3, 512, 512]), "interpolation"),
]


@pytest.mark.parametrize("tile_size, stride", [(512, 256), ([512, 512], [256, 256]), (ListConfig([512, 512]), 256)])
def test_size_types_should_be_int_tuple_or_list_config(tile_size, stride):
    tiler = Tiler(tile_size=tile_size, stride=stride)
    assert isinstance(tiler.tile_size_h, int)
    assert isinstance(tiler.stride_w, int)


@pytest.mark.parametrize("image_size, tile_size, stride, shape", tile_data)
def test_tiler_handles_single_image_without_batch_dimension(image_size, tile_size, stride, shape):
    tiler = Tiler(tile_size=tile_size, stride=stride)
    image = torch.rand(image_size)
    patches = tiler.tile(image)
    assert patches.shape == shape


def test_overlapping_disabled_when_kernel_size_equals_to_stride():
    tiler = Tiler(tile_size=512, stride=512)
    assert tiler.overlapping == False


def test_stride_size_cannot_be_larger_than_tile_size():
    kernel_size = (128, 128)
    stride = 256
    with pytest.raises(StrideSizeError):
        tiler = Tiler(tile_size=kernel_size, stride=stride)


@pytest.mark.parametrize("tile_size, kernel_size, stride, image_size", untile_data)
def test_untile_non_overlapping_patches(tile_size, kernel_size, stride, image_size):
    tiler = Tiler(tile_size=kernel_size, stride=stride)
    image = torch.rand(image_size)
    tiles = tiler.tile(image)

    untiled_image = tiler.untile(tiles)
    assert untiled_image.shape == torch.Size(image_size)


@pytest.mark.parametrize("image_size, kernel_size, stride, tile_size, mode", overlapping_data)
def test_untile_overlapping_patches(image_size, kernel_size, stride, tile_size, mode):
    tiler = Tiler(tile_size=kernel_size, stride=stride, mode=mode)

    image = torch.rand(image_size)
    tiles = tiler.tile(image)
    reconstructed_image = tiler.untile(tiles)
    assert torch.equal(image, reconstructed_image)


@pytest.mark.parametrize(
    "image_size, tile_size, stride, mode",
    [
        ([1, 3, 1024, 1024], [211, 210], [197, 197], "padding"),
        ([2, 3, 512, 512], [120, 124], [100, 100], "interpolation"),
    ],
)
def test_non_divisible_tile_size_and_stride_should_be_handled(image_size, tile_size, stride, mode):

    tiler = Tiler(tile_size=tile_size, stride=stride, mode=mode)

    image = torch.rand(image_size)
    tiles = tiler.tile(image)
    reconstructed_image = tiler.untile(tiles)
    assert image.shape == reconstructed_image.shape
