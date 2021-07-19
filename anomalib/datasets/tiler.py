"""
Image Tiler
"""

from math import ceil
from typing import Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F


class StrideSizeError(Exception):
    """
    StrideSizeError to raise exception when stride
        size is greater than the tile size.
    """


def compute_new_image_size(image_size: Tuple, tile_size: Tuple, stride: Tuple) -> Tuple:
    """
    This function checks if image size is divisible by tile size and stride.
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
        """
        This function makes the resizing within the edge level.
        """
        if (edge_size - tile_size) % stride != 0:
            edge_size = (ceil((edge_size - tile_size) / stride) * stride) + tile_size

        return edge_size

    resized_h = __compute_new_edge_size(image_size[0], tile_size[0], stride[0])
    resized_w = __compute_new_edge_size(image_size[1], tile_size[1], stride[1])

    return (resized_h, resized_w)


def upscale_image(image: Tensor, size: Tuple, mode: str = "padding") -> Tensor:
    """
    Upscale image to the desired size via either padding or interpolation.

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
    """
    Opposite of upscaling. This image downscales image to a desired size.

    Args:
        image (Tensor): Input image
        size (Tuple): Size to which image is down scaled.
        mode (str, optional): Downscaling mode. Defaults to "padding".

    Examples:
        >>> x = torch.rand(1, 3, 512, 512)
        >>> y = upscale_image(image, upscale_size=(555, 555), mode="padding")
        >>> y = downscale_image(image, size=(512, 512), mode='padding')
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
    """
    Tile Image into (non)overlapping Patches

    Examples:
        >>> import torch
        >>> from torchvision import transforms
        >>> from skimage.data import camera
        >>> tiler = Tiler(tile_size=256, stride=128)
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
        mode: str = "padding",
    ) -> None:

        self.tile_size_h, self.tile_size_w = self.__validate_size_type(tile_size)
        self.stride_h, self.stride_w = self.__validate_size_type(stride)
        self.overlapping = False if (self.stride_h == self.tile_size_h and self.stride_w == self.tile_size_w) else True
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
    def __validate_size_type(parameter) -> Tuple:
        if isinstance(parameter, int):
            output = (parameter,) * 2
        elif isinstance(parameter, Sequence):
            output = tuple(parameter)
        else:
            raise ValueError(f"Unknown type {type(parameter)} for tile or stride size. Could be int or Sequence type.")

        if len(output) != 2:
            raise ValueError(f"Length of the size type must be 2 for height and width. Got {len(output)} instead.")

        return output

    def __unfold(self, tensor: Tensor) -> Tensor:
        """
        Unfolds tensor into tiles. This is the core function to perform tiling operation.

        Args:
            tensor: Input tensor from which tiles are generated.

        Returns: Generated tiles

        """
        tiles = tensor.unfold(2, self.tile_size_h, self.stride_h).unfold(3, self.tile_size_w, self.stride_w)

        self.num_patches_h, self.num_patches_w = tiles.shape[2:4]
        # [batch, num_patch_h, num_patch_w, num_channel, tile_size, tile_size]
        tiles = tiles.permute(0, 2, 3, 1, 4, 5)
        # [batch * num patches, kernel size, kernel size]
        tiles = tiles.contiguous().view(-1, self.num_channels, self.tile_size_h, self.tile_size_w)

        return tiles

    def __fold(self, tiles: Tensor) -> Tensor:
        """
        Fold the tiles back into the original tensor. This is the core method to reconstruct
        the original image from its tiled version.

        Args:
            tiles: Tiles from the input image, generated via __unfold method.

        Returns:
            Output that is the reconstructed version of the input tensor.

        """
        num_channels = tiles.shape[1]
        tiles = tiles.contiguous().view(
            self.batch_size,
            self.num_patches_h,
            self.num_patches_w,
            num_channels,
            self.tile_size_h,
            self.tile_size_w,
        )
        tiles = tiles.permute(0, 3, 1, 2, 4, 5)
        tiles = tiles.contiguous().view(self.batch_size, num_channels, -1, self.tile_size_h * self.tile_size_w)
        tiles = tiles.permute(0, 1, 3, 2)
        tiles = tiles.contiguous().view(self.batch_size, num_channels * self.tile_size_h * self.tile_size_w, -1)

        image = F.fold(
            tiles,
            output_size=(self.resized_h, self.resized_w),
            kernel_size=(self.tile_size_h, self.tile_size_w),
            stride=(self.stride_h, self.stride_w),
        )
        return image

    def tile(self, image: Tensor) -> Tensor:
        """
        Tiles an input image to either overlapping or non-overlapping patches.

        Args:
            image: Input image to tile.

        Examples:
            >>> from anomalib.datasets.tiler import Tiler
            >>> tiler = Tiler(tile_size=512, stride=256)
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

        image_tiles = self.__unfold(image)
        return image_tiles

    def untile(self, tiles: Tensor) -> Tensor:
        """
        Untiles patches to reconstruct the original input image. If patches, are overlapping
        patches, the function averages the overlapping pixels, and return the reconstructed
        image.

        Args:
            tiles: Tiles from the input image, generated via tile()..

        Examples:

            >>> from anomalib.datasets.tiler import Tiler
            >>> tiler = Tiler(tile_size=512, stride=256)
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

        if self.overlapping:
            mask = torch.ones(image.shape)
            tiled_mask = self.__unfold(mask)
            untiled_mask = self.__fold(tiled_mask)
            image = torch.div(image, untiled_mask)

        image = downscale_image(image=image, size=(self.input_h, self.input_w), mode=self.mode)

        return image
