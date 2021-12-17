"""Image Tiler."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from itertools import product
from math import ceil
from typing import Optional, Sequence, SupportsIndex, Tuple, Union

import torch
import torchvision.transforms as T
from torch import Tensor
from torch.nn import functional as F


class StrideSizeError(Exception):
    """StrideSizeError to raise exception when stride size is greater than the tile size."""


def compute_new_image_size(image_size: Tuple, tile_size: Tuple, stride: Tuple) -> Tuple:
    """This function checks if image size is divisible by tile size and stride.

    If not divisible, it resizes the image size to make it divisible.

    Args:
        image_size (Tuple): Original image size
        tile_size (Tuple): Tile size
        stride (Tuple): Stride

    Examples:
        >>> compute_new_image_size(image_size=(512, 512), tile_size=(256, 256), stride=(128, 128))
        (512, 512)

        >>> compute_new_image_size(image_size=(512, 512), tile_size=(222, 222), stride=(111, 111))
        (555, 555)

    Returns:
        Tuple: Updated image size that is divisible by tile size and stride.
    """

    def __compute_new_edge_size(edge_size: int, tile_size: int, stride: int) -> int:
        """This function makes the resizing within the edge level."""
        if (edge_size - tile_size) % stride != 0:
            edge_size = (ceil((edge_size - tile_size) / stride) * stride) + tile_size

        return edge_size

    resized_h = __compute_new_edge_size(image_size[0], tile_size[0], stride[0])
    resized_w = __compute_new_edge_size(image_size[1], tile_size[1], stride[1])

    return resized_h, resized_w


def upscale_image(image: Tensor, size: Tuple, mode: str = "padding") -> Tensor:
    """Upscale image to the desired size via either padding or interpolation.

    Args:
        image (Tensor): Image
        size (Tuple): Tuple to which image is upscaled.
        mode (str, optional): Upscaling mode. Defaults to "padding".

    Examples:
        >>> image = torch.rand(1, 3, 512, 512)
        >>> image = upscale_image(image, size=(555, 555), mode="padding")
        >>> image.shape
        torch.Size([1, 3, 555, 555])

        >>> image = torch.rand(1, 3, 512, 512)
        >>> image = upscale_image(image, size=(555, 555), mode="interpolation")
        >>> image.shape
        torch.Size([1, 3, 555, 555])

    Returns:
        Tensor: Upscaled image.
    """

    image_h, image_w = image.shape[2:]
    resize_h, resize_w = size

    if mode == "padding":
        pad_h = resize_h - image_h
        pad_w = resize_w - image_w

        image = F.pad(image, [0, pad_w, 0, pad_h])
    elif mode == "interpolation":
        image = F.interpolate(input=image, size=(resize_h, resize_w))
    else:
        raise ValueError(f"Unknown mode {mode}. Only padding and interpolation is available.")

    return image


def downscale_image(image: Tensor, size: Tuple, mode: str = "padding") -> Tensor:
    """Opposite of upscaling. This image downscales image to a desired size.

    Args:
        image (Tensor): Input image
        size (Tuple): Size to which image is down scaled.
        mode (str, optional): Downscaling mode. Defaults to "padding".

    Examples:
        >>> x = torch.rand(1, 3, 512, 512)
        >>> y = upscale_image(image, upscale_size=(555, 555), mode="padding")
        >>> y = downscale_image(y, size=(512, 512), mode='padding')
        >>> torch.allclose(x, y)
        True

    Returns:
        Tensor: Downscaled image
    """
    input_h, input_w = size
    if mode == "padding":
        image = image[:, :, :input_h, :input_w]
    else:
        image = F.interpolate(input=image, size=(input_h, input_w))

    return image


class Tiler:
    """Tile Image into (non)overlapping Patches. Images are tiled in order to efficiently process large images.

    Args:
        tile_size: Tile dimension for each patch
        stride: Stride length between patches
        remove_border_count: Number of border pixels to be removed from tile before untiling
        mode: Upscaling mode for image resize.Supported formats: padding, interpolation

    Examples:
        >>> import torch
        >>> from torchvision import transforms
        >>> from skimage.data import camera
        >>> tiler = Tiler(tile_size=256,stride=128)
        >>> image = transforms.ToTensor()(camera())
        >>> tiles = tiler.tile(image)
        >>> image.shape, tiles.shape
        (torch.Size([3, 512, 512]), torch.Size([9, 3, 256, 256]))

        >>> # Perform your operations on the tiles.

        >>> # Untile the patches to reconstruct the image
        >>> reconstructed_image = tiler.untile(tiles)
        >>> reconstructed_image.shape
        torch.Size([1, 3, 512, 512])
    """

    def __init__(
        self,
        tile_size: Union[int, Sequence],
        stride: Union[int, Sequence],
        remove_border_count: int = 0,
        mode: str = "padding",
        tile_count: SupportsIndex = 4,
    ) -> None:

        self.tile_size_h, self.tile_size_w = self.__validate_size_type(tile_size)
        self.tile_count = tile_count
        self.stride_h, self.stride_w = self.__validate_size_type(stride)
        self.remove_border_count = int(remove_border_count)
        self.overlapping = not (self.stride_h == self.tile_size_h and self.stride_w == self.tile_size_w)
        self.mode = mode

        if self.stride_h > self.tile_size_h or self.stride_w > self.tile_size_w:
            raise StrideSizeError(
                "Larger stride size than kernel size produces unreliable tiling results. "
                "Please ensure stride size is less than or equal than tiling size."
            )

        if self.mode not in ["padding", "interpolation"]:
            raise ValueError(f"Unknown tiling mode {self.mode}. Available modes are padding and interpolation")

        self.batch_size: int
        self.num_channels: int

        self.input_h: int
        self.input_w: int

        self.pad_h: int
        self.pad_w: int

        self.resized_h: int
        self.resized_w: int

        self.num_patches_h: int
        self.num_patches_w: int

    @staticmethod
    def __validate_size_type(parameter) -> Tuple[int, int]:
        if isinstance(parameter, int):
            output = (parameter, parameter)
        elif isinstance(parameter, Sequence):
            output = (parameter[0], parameter[1])
        else:
            raise ValueError(f"Unknown type {type(parameter)} for tile or stride size. Could be int or Sequence type.")

        if len(output) != 2:
            raise ValueError(f"Length of the size type must be 2 for height and width. Got {len(output)} instead.")

        return output

    def __random_tile(self, image: Tensor) -> Tensor:
        """Randomly crop tiles from the given image.

        Args:
            image: input image to be cropped

        Returns: Randomly cropped tiles from the image
        """
        return torch.vstack([T.RandomCrop(self.tile_size_h)(image) for i in range(self.tile_count)])

    def __unfold(self, tensor: Tensor) -> Tensor:
        """Unfolds tensor into tiles.

        This is the core function to perform tiling operation.

        Args:
            tensor: Input tensor from which tiles are generated.

        Returns: Generated tiles
        """

        # identify device type based on input tensor
        device = tensor.device

        # extract and calculate parameters
        batch, channels, image_h, image_w = tensor.shape

        self.num_patches_h = int((image_h - self.tile_size_h) / self.stride_h) + 1
        self.num_patches_w = int((image_w - self.tile_size_w) / self.stride_w) + 1

        # create an empty torch tensor for output
        tiles = torch.zeros(
            (self.num_patches_h, self.num_patches_w, batch, channels, self.tile_size_h, self.tile_size_w), device=device
        )

        # fill-in output tensor with spatial patches extracted from the image
        for (tile_i, tile_j), (loc_i, loc_j) in zip(
            product(range(self.num_patches_h), range(self.num_patches_w)),
            product(
                range(0, image_h - self.tile_size_h + 1, self.stride_h),
                range(0, image_w - self.tile_size_w + 1, self.stride_w),
            ),
        ):
            tiles[tile_i, tile_j, :] = tensor[
                :, :, loc_i : (loc_i + self.tile_size_h), loc_j : (loc_j + self.tile_size_w)
            ]

        # rearrange the tiles in order [tile_count * batch, channels, tile_height, tile_width]
        tiles = tiles.permute(2, 0, 1, 3, 4, 5)
        tiles = tiles.contiguous().view(-1, channels, self.tile_size_h, self.tile_size_w)

        return tiles

    def __fold(self, tiles: Tensor) -> Tensor:
        """Fold the tiles back into the original tensor.

        This is the core method to reconstruct the original image from its tiled version.

        Args:
            tiles: Tiles from the input image, generated via __unfold method.

        Returns:
            Output that is the reconstructed version of the input tensor.
        """
        # number of channels differs between image and anomaly map, so infer from input tiles.
        _, num_channels, tile_size_h, tile_size_w = tiles.shape
        scale_h, scale_w = (tile_size_h / self.tile_size_h), (tile_size_w / self.tile_size_w)
        # identify device type based on input tensor
        device = tiles.device
        # calculate tile size after borders removed
        reduced_tile_h = tile_size_h - (2 * self.remove_border_count)
        reduced_tile_w = tile_size_w - (2 * self.remove_border_count)
        # reconstructed image dimension
        image_size = (self.batch_size, num_channels, int(self.resized_h * scale_h), int(self.resized_w * scale_w))

        # rearrange input tiles in format [tile_count, batch, channel, tile_h, tile_w]
        tiles = tiles.contiguous().view(
            self.batch_size,
            self.num_patches_h,
            self.num_patches_w,
            num_channels,
            tile_size_h,
            tile_size_w,
        )
        tiles = tiles.permute(0, 3, 1, 2, 4, 5)
        tiles = tiles.contiguous().view(self.batch_size, num_channels, -1, tile_size_h, tile_size_w)
        tiles = tiles.permute(2, 0, 1, 3, 4)

        # remove tile borders by defined count
        tiles = tiles[
            :,
            :,
            :,
            self.remove_border_count : reduced_tile_h + self.remove_border_count,
            self.remove_border_count : reduced_tile_w + self.remove_border_count,
        ]

        # create tensors to store intermediate results and outputs
        img = torch.zeros(image_size, device=device)
        lookup = torch.zeros(image_size, device=device)
        ones = torch.ones(reduced_tile_h, reduced_tile_w, device=device)

        # reconstruct image by adding patches to their respective location and
        # create a lookup for patch count in every location
        for patch, (loc_i, loc_j) in zip(
            tiles,
            product(
                range(
                    self.remove_border_count,
                    int(self.resized_h * scale_h) - reduced_tile_h + 1,
                    int(self.stride_h * scale_h),
                ),
                range(
                    self.remove_border_count,
                    int(self.resized_w * scale_w) - reduced_tile_w + 1,
                    int(self.stride_w * scale_w),
                ),
            ),
        ):
            img[:, :, loc_i : (loc_i + reduced_tile_h), loc_j : (loc_j + reduced_tile_w)] += patch
            lookup[:, :, loc_i : (loc_i + reduced_tile_h), loc_j : (loc_j + reduced_tile_w)] += ones

        # divide the reconstucted image by the lookup to average out the values
        img = torch.divide(img, lookup)
        # alternative way of removing nan values (isnan not supported by openvino)
        img[img != img] = 0  # pylint: disable=comparison-with-itself

        return img

    def tile(self, image: Tensor, use_random_tiling: Optional[bool] = False) -> Tensor:
        """Tiles an input image to either overlapping, non-overlapping or random patches.

        Args:
            image: Input image to tile.

        Examples:
            >>> from anomalib.data.tiler import Tiler
            >>> tiler = Tiler(tile_size=512,stride=256)
            >>> image = torch.rand(size=(2, 3, 1024, 1024))
            >>> image.shape
            torch.Size([2, 3, 1024, 1024])
            >>> tiles = tiler.tile(image)
            >>> tiles.shape
            torch.Size([18, 3, 512, 512])

        Returns:
            Tiles generated from the image.
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        self.batch_size, self.num_channels, self.input_h, self.input_w = image.shape

        if self.input_h < self.tile_size_h or self.input_w < self.tile_size_w:
            raise ValueError(
                f"One of the edges of the tile size {self.tile_size_h, self.tile_size_w} "
                "is larger than that of the image {self.input_h, self.input_w}."
            )

        self.resized_h, self.resized_w = compute_new_image_size(
            image_size=(self.input_h, self.input_w),
            tile_size=(self.tile_size_h, self.tile_size_w),
            stride=(self.stride_h, self.stride_w),
        )

        image = upscale_image(image, size=(self.resized_h, self.resized_w), mode=self.mode)

        if use_random_tiling:
            image_tiles = self.__random_tile(image)
        else:
            image_tiles = self.__unfold(image)
        return image_tiles

    def untile(self, tiles: Tensor) -> Tensor:
        """Untiles patches to reconstruct the original input image.

        If patches, are overlapping patches, the function averages the overlapping pixels,
        and return the reconstructed image.

        Args:
            tiles: Tiles from the input image, generated via tile()..

        Examples:
            >>> from anomalib.datasets.tiler import Tiler
            >>> tiler = Tiler(tile_size=512,stride=256)
            >>> image = torch.rand(size=(2, 3, 1024, 1024))
            >>> image.shape
            torch.Size([2, 3, 1024, 1024])
            >>> tiles = tiler.tile(image)
            >>> tiles.shape
            torch.Size([18, 3, 512, 512])
            >>> reconstructed_image = tiler.untile(tiles)
            >>> reconstructed_image.shape
            torch.Size([2, 3, 1024, 1024])
            >>> torch.equal(image, reconstructed_image)
            True

        Returns:
            Output that is the reconstructed version of the input tensor.
        """
        image = self.__fold(tiles)
        image = downscale_image(image=image, size=(self.input_h, self.input_w), mode=self.mode)

        return image
