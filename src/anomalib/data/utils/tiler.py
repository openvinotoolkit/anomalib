"""Image tiling utilities for processing large images.

This module provides functionality to:

- Tile large images into smaller patches for efficient processing
- Support overlapping and non-overlapping tiling strategies
- Reconstruct original images from tiles
- Handle upscaling and downscaling with padding or interpolation

Example:
    >>> from anomalib.data.utils.tiler import Tiler
    >>> import torch
    >>> # Create tiler with 256x256 tiles and 128 stride
    >>> tiler = Tiler(tile_size=256, stride=128)
    >>> # Create sample 512x512 image
    >>> image = torch.rand(1, 3, 512, 512)
    >>> # Generate tiles
    >>> tiles = tiler.tile(image)
    >>> tiles.shape
    torch.Size([9, 3, 256, 256])
    >>> # Reconstruct image from tiles
    >>> reconstructed = tiler.untile(tiles)
    >>> reconstructed.shape
    torch.Size([1, 3, 512, 512])
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from enum import Enum
from itertools import product
from math import ceil

import torch
import torchvision.transforms as T  # noqa: N812
from torch.nn import functional as F  # noqa: N812


class ImageUpscaleMode(str, Enum):
    """Mode for upscaling images.

    Attributes:
        PADDING: Upscale by padding with zeros
        INTERPOLATION: Upscale using interpolation
    """

    PADDING = "padding"
    INTERPOLATION = "interpolation"


class StrideSizeError(Exception):
    """Error raised when stride size exceeds tile size."""


def compute_new_image_size(image_size: tuple, tile_size: tuple, stride: tuple) -> tuple:
    """Compute new image size that is divisible by tile size and stride.

    Args:
        image_size: Original image size as ``(height, width)``
        tile_size: Tile size as ``(height, width)``
        stride: Stride size as ``(height, width)``

    Returns:
        tuple: New image size divisible by tile size and stride

    Examples:
        >>> compute_new_image_size((512, 512), (256, 256), (128, 128))
        (512, 512)
        >>> compute_new_image_size((512, 512), (222, 222), (111, 111))
        (555, 555)
    """

    def __compute_new_edge_size(edge_size: int, tile_size: int, stride: int) -> int:
        """Compute new edge size that is divisible by tile size and stride.

        Args:
            edge_size: Original edge size
            tile_size: Tile size for this edge
            stride: Stride size for this edge

        Returns:
            int: New edge size
        """
        if (edge_size - tile_size) % stride != 0:
            edge_size = (ceil((edge_size - tile_size) / stride) * stride) + tile_size

        return edge_size

    resized_h = __compute_new_edge_size(image_size[0], tile_size[0], stride[0])
    resized_w = __compute_new_edge_size(image_size[1], tile_size[1], stride[1])

    return resized_h, resized_w


def upscale_image(
    image: torch.Tensor,
    size: tuple,
    mode: ImageUpscaleMode = ImageUpscaleMode.PADDING,
) -> torch.Tensor:
    """Upscale image to desired size using padding or interpolation.

    Args:
        image: Input image tensor
        size: Target size as ``(height, width)``
        mode: Upscaling mode, either ``"padding"`` or ``"interpolation"``

    Returns:
        torch.Tensor: Upscaled image

    Examples:
        >>> image = torch.rand(1, 3, 512, 512)
        >>> upscaled = upscale_image(image, (555, 555), "padding")
        >>> upscaled.shape
        torch.Size([1, 3, 555, 555])

        >>> upscaled = upscale_image(image, (555, 555), "interpolation")
        >>> upscaled.shape
        torch.Size([1, 3, 555, 555])
    """
    image_h, image_w = image.shape[2:]
    resize_h, resize_w = size

    if mode == ImageUpscaleMode.PADDING:
        pad_h = resize_h - image_h
        pad_w = resize_w - image_w

        image = F.pad(image, [0, pad_w, 0, pad_h])
    elif mode == ImageUpscaleMode.INTERPOLATION:
        image = F.interpolate(input=image, size=(resize_h, resize_w))
    else:
        msg = f"Unknown mode {mode}. Only padding and interpolation is available."
        raise ValueError(msg)

    return image


def downscale_image(
    image: torch.Tensor,
    size: tuple,
    mode: ImageUpscaleMode = ImageUpscaleMode.PADDING,
) -> torch.Tensor:
    """Downscale image to desired size.

    Args:
        image: Input image tensor
        size: Target size as ``(height, width)``
        mode: Downscaling mode, either ``"padding"`` or ``"interpolation"``

    Returns:
        torch.Tensor: Downscaled image

    Examples:
        >>> x = torch.rand(1, 3, 512, 512)
        >>> y = upscale_image(x, (555, 555), "padding")
        >>> z = downscale_image(y, (512, 512), "padding")
        >>> torch.allclose(x, z)
        True
    """
    input_h, input_w = size
    if mode == ImageUpscaleMode.PADDING:
        image = image[:, :, :input_h, :input_w]
    else:
        image = F.interpolate(input=image, size=(input_h, input_w))

    return image


class Tiler:
    """Tile images into overlapping or non-overlapping patches.

    This class provides functionality to:
    - Split large images into smaller tiles for efficient processing
    - Support overlapping tiles with configurable stride
    - Remove border pixels from tiles before reconstruction
    - Reconstruct original image from processed tiles

    Args:
        tile_size: Size of tiles as int or ``(height, width)``
        stride: Stride between tiles as int or ``(height, width)``.
            If ``None``, uses tile_size (non-overlapping)
        remove_border_count: Number of border pixels to remove from tiles
        mode: Upscaling mode for resizing, either ``"padding"`` or
            ``"interpolation"``

    Examples:
        >>> import torch
        >>> from torchvision import transforms
        >>> from skimage.data import camera
        >>> # Create tiler for 256x256 tiles with 128 stride
        >>> tiler = Tiler(tile_size=256, stride=128)
        >>> # Convert test image to tensor
        >>> image = transforms.ToTensor()(camera())
        >>> # Generate tiles
        >>> tiles = tiler.tile(image)
        >>> image.shape, tiles.shape
        (torch.Size([3, 512, 512]), torch.Size([9, 3, 256, 256]))

        >>> # Process tiles here...

        >>> # Reconstruct image from tiles
        >>> reconstructed = tiler.untile(tiles)
        >>> reconstructed.shape
        torch.Size([1, 3, 512, 512])
    """

    def __init__(
        self,
        tile_size: int | Sequence,
        stride: int | Sequence | None = None,
        remove_border_count: int = 0,
        mode: ImageUpscaleMode = ImageUpscaleMode.PADDING,
    ) -> None:
        self.tile_size_h, self.tile_size_w = self.validate_size_type(tile_size)
        self.random_tile_count = 4

        if stride is not None:
            self.stride_h, self.stride_w = self.validate_size_type(stride)

        self.remove_border_count = remove_border_count
        self.overlapping = not (self.stride_h == self.tile_size_h and self.stride_w == self.tile_size_w)
        self.mode = mode

        if self.stride_h > self.tile_size_h or self.stride_w > self.tile_size_w:
            msg = "Stride size larger than tile size produces unreliable results. Ensure stride size <= tile size."
            raise StrideSizeError(msg)

        if self.mode not in {ImageUpscaleMode.PADDING, ImageUpscaleMode.INTERPOLATION}:
            msg = f"Unknown mode {self.mode}. Available modes: padding and interpolation"
            raise ValueError(msg)

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
    def validate_size_type(parameter: int | Sequence) -> tuple[int, ...]:
        """Validate and convert size parameter to tuple.

        Args:
            parameter: Size as int or sequence of ``(height, width)``

        Returns:
            tuple: Validated size as ``(height, width)``

        Raises:
            TypeError: If parameter type is invalid
            ValueError: If parameter length is not 2
        """
        if isinstance(parameter, int):
            output = (parameter, parameter)
        elif isinstance(parameter, Sequence):
            output = (parameter[0], parameter[1])
        else:
            msg = f"Invalid type {type(parameter)} for tile/stride size. Must be int or Sequence."
            raise TypeError(msg)

        if len(output) != 2:
            msg = f"Size must have length 2, got {len(output)}"
            raise ValueError(msg)

        return output

    def __random_tile(self, image: torch.Tensor) -> torch.Tensor:
        """Randomly crop tiles from image.

        Args:
            image: Input image tensor

        Returns:
            torch.Tensor: Stack of random tiles
        """
        return torch.vstack([T.RandomCrop(self.tile_size_h)(image) for i in range(self.random_tile_count)])

    def __unfold(self, tensor: torch.Tensor) -> torch.Tensor:
        """Unfold tensor into tiles.

        Args:
            tensor: Input tensor to tile

        Returns:
            torch.Tensor: Generated tiles
        """
        device = tensor.device
        batch, channels, image_h, image_w = tensor.shape

        self.num_patches_h = int((image_h - self.tile_size_h) / self.stride_h) + 1
        self.num_patches_w = int((image_w - self.tile_size_w) / self.stride_w) + 1

        tiles = torch.zeros(
            (
                self.num_patches_h,
                self.num_patches_w,
                batch,
                channels,
                self.tile_size_h,
                self.tile_size_w,
            ),
            device=device,
        )

        for (tile_i, tile_j), (loc_i, loc_j) in zip(
            product(range(self.num_patches_h), range(self.num_patches_w)),
            product(
                range(0, image_h - self.tile_size_h + 1, self.stride_h),
                range(0, image_w - self.tile_size_w + 1, self.stride_w),
            ),
            strict=True,
        ):
            tiles[tile_i, tile_j, :] = tensor[
                :,
                :,
                loc_i : (loc_i + self.tile_size_h),
                loc_j : (loc_j + self.tile_size_w),
            ]

        tiles = tiles.permute(2, 0, 1, 3, 4, 5)
        return tiles.contiguous().view(-1, channels, self.tile_size_h, self.tile_size_w)

    def __fold(self, tiles: torch.Tensor) -> torch.Tensor:
        """Fold tiles back into original tensor.

        Args:
            tiles: Tiles generated by ``__unfold()``

        Returns:
            torch.Tensor: Reconstructed tensor
        """
        _, num_channels, tile_size_h, tile_size_w = tiles.shape
        scale_h, scale_w = (tile_size_h / self.tile_size_h), (tile_size_w / self.tile_size_w)
        device = tiles.device
        reduced_tile_h = tile_size_h - (2 * self.remove_border_count)
        reduced_tile_w = tile_size_w - (2 * self.remove_border_count)
        image_size = (
            self.batch_size,
            num_channels,
            int(self.resized_h * scale_h),
            int(self.resized_w * scale_w),
        )

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

        tiles = tiles[
            :,
            :,
            :,
            self.remove_border_count : reduced_tile_h + self.remove_border_count,
            self.remove_border_count : reduced_tile_w + self.remove_border_count,
        ]

        img = torch.zeros(image_size, device=device)
        lookup = torch.zeros(image_size, device=device)
        ones = torch.ones(reduced_tile_h, reduced_tile_w, device=device)

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
            strict=True,
        ):
            img[
                :,
                :,
                loc_i : (loc_i + reduced_tile_h),
                loc_j : (loc_j + reduced_tile_w),
            ] += patch
            lookup[
                :,
                :,
                loc_i : (loc_i + reduced_tile_h),
                loc_j : (loc_j + reduced_tile_w),
            ] += ones

        img = torch.divide(img, lookup)
        img[img != img] = 0  # noqa: PLR0124

        return img

    def tile(self, image: torch.Tensor, use_random_tiling: bool = False) -> torch.Tensor:
        """Tile input image into patches.

        Args:
            image: Input image tensor
            use_random_tiling: If ``True``, randomly crop tiles.
                If ``False``, tile in regular grid.

        Returns:
            torch.Tensor: Generated tiles

        Examples:
            >>> tiler = Tiler(tile_size=512, stride=256)
            >>> image = torch.rand(2, 3, 1024, 1024)
            >>> tiles = tiler.tile(image)
            >>> tiles.shape
            torch.Size([18, 3, 512, 512])

        Raises:
            ValueError: If tile size exceeds image size
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        self.batch_size, self.num_channels, self.input_h, self.input_w = image.shape

        if self.input_h < self.tile_size_h or self.input_w < self.tile_size_w:
            msg = f"Tile size {self.tile_size_h, self.tile_size_w} exceeds image size {self.input_h, self.input_w}"
            raise ValueError(msg)

        self.resized_h, self.resized_w = compute_new_image_size(
            image_size=(self.input_h, self.input_w),
            tile_size=(self.tile_size_h, self.tile_size_w),
            stride=(self.stride_h, self.stride_w),
        )

        image = upscale_image(image, size=(self.resized_h, self.resized_w), mode=self.mode)

        return self.__random_tile(image) if use_random_tiling else self.__unfold(image)

    def untile(self, tiles: torch.Tensor) -> torch.Tensor:
        """Reconstruct image from tiles.

        For overlapping tiles, averages overlapping regions.

        Args:
            tiles: Tiles generated by ``tile()``

        Returns:
            torch.Tensor: Reconstructed image

        Examples:
            >>> tiler = Tiler(tile_size=512, stride=256)
            >>> image = torch.rand(2, 3, 1024, 1024)
            >>> tiles = tiler.tile(image)
            >>> reconstructed = tiler.untile(tiles)
            >>> reconstructed.shape
            torch.Size([2, 3, 1024, 1024])
            >>> torch.equal(image, reconstructed)
            True
        """
        image = self.__fold(tiles)
        return downscale_image(image=image, size=(self.input_h, self.input_w), mode=self.mode)
