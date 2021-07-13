"""
Dataset Utils
"""

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import ListConfig
from torch import Tensor
from torch.nn import functional as F


def resize_edge(image_size: int, kernel_size: int, stride: int) -> int:
    # return round(image_size / kernel_size) * kernel_size
    # return (ceil((image_size - kernel_size) / stride) + 1) * kernel_size
    return image_size // stride * stride + kernel_size


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

    # def __init__(self, tile_size: int, stride: int) -> None:
    def __init__(self, tile_size: Union[int, ListConfig, Tuple], stride: Union[int, ListConfig, Tuple]) -> None:

        self.tile_size_h, self.tile_size_w = self.__validate_size_type(tile_size)
        self.stride_h, self.stride_w = self.__validate_size_type(stride)
        self.overlapping = False if (self.stride_h == self.tile_size_h and self.stride_w == self.tile_size_w) else True

        if self.stride_h > self.tile_size_h:
            warnings.warn(
                message="Height of the stride is greater than height of the tile size. "
                "This could cause unreliable tiling issues. Hence setting height of the stride to tiling."
            )
            self.stride_h = self.tile_size_h

        if self.stride_w > self.tile_size_w:
            warnings.warn(
                message="Width of the stride is greater than width of the tile size. "
                "This could cause unreliable tiling issues. Hence setting width of the stride to tiling."
            )
            self.stride_w = self.tile_size_w

        self.batch_size: int
        self.num_channels: int

        self.input_h: int
        self.input_w: int

        self.resized_h: int
        self.resized_w: int

        self.num_patches_h: int
        self.num_patches_w: int

    @staticmethod
    def __validate_size_type(parameter) -> Tuple:
        if isinstance(parameter, tuple):
            output = parameter
        else:
            if isinstance(parameter, int):
                output = (parameter,) * 2
            elif isinstance(parameter, ListConfig):
                output = tuple(parameter)
            else:
                raise ValueError(
                    f"Unknown type {type(parameter)} for tile or stride size. Could be int, tuple or ListConfig"
                )
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
        tiles = tiles.contiguous().view(
            self.batch_size,
            self.num_patches_h,
            self.num_patches_w,
            self.num_channels,
            self.tile_size_h,
            self.tile_size_w,
        )
        tiles = tiles.permute(0, 3, 1, 2, 4, 5)
        tiles = tiles.contiguous().view(self.batch_size, self.num_channels, -1, self.tile_size_h * self.tile_size_w)
        tiles = tiles.permute(0, 1, 3, 2)
        tiles = tiles.contiguous().view(self.batch_size, self.num_channels * self.tile_size_h * self.tile_size_w, -1)
        image = F.fold(
            tiles,
            output_size=(self.input_h, self.input_w),
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
            >>> from anomalib.datasets.utils import Tiler
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
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        self.batch_size, self.num_channels, self.input_h, self.input_w = image.shape

        # # Resize the image if tile size is not divisable.
        # self.resized_h = resize_edge(self.input_h, self.tile_size_h, self.stride_h)
        # self.resized_w = resize_edge(self.input_w, self.tile_size_w, self.stride_w)
        # image = F.interpolate(input=image, size=(self.resized_h, self.resized_w))

        image_patches = self.__unfold(image)
        return image_patches

    def untile(self, tiles: Tensor) -> Tensor:
        """
        Untiles patches to reconstruct the original input image. If patches, are overlapping
        patches, the function averages the overlapping pixels, and return the reconstructed
        image.

        Args:
            tiles: Tiles from the input image, generated via tile()..

        Examples:

            >>> from anomalib.datasets.utils import Tiler
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

        mask = torch.ones(image.shape)
        tiled_mask = self.__unfold(mask)
        untiled_mask = self.__fold(tiled_mask)

        image = torch.div(image, untiled_mask)
        # image = F.interpolate(input=image, size=(self.input_h, self.input_w))

        return image


class Denormalize:
    """
    Denormalize Torch Tensor into np image format.
    """

    def __init__(self, mean: Optional[List[float]] = None, std: Optional[List[float]] = None):
        # If no mean and std provided, assign ImageNet values.
        if mean is None:
            mean = [0.485, 0.456, 0.406]

        if std is None:
            std = [0.229, 0.224, 0.225]

        self.mean = Tensor(mean)
        self.std = Tensor(std)

    def __call__(self, tensor: Tensor) -> np.ndarray:
        """
        Denormalize the input

        Args:
            tensor: Input tensor image (C, H, W)

        Returns:
            Denormalized numpy array (H, W, C).

        """
        for tnsr, mean, std in zip(tensor, self.mean, self.std):
            tnsr.mul_(std).add_(mean)

        array = (tensor * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return array

    def __repr__(self):
        return self.__class__.__name__ + "()"


class ToNumpy:
    """
    Convert Tensor into Numpy Array
    """

    def __call__(self, tensor: Tensor, dims: Optional[Tuple[int, ...]] = None) -> np.ndarray:

        # Default support is (C, H, W) or (N, C, H, W)
        if dims is None:
            dims = (0, 2, 3, 1) if len(tensor.shape) == 4 else (1, 2, 0)

        array = (tensor * 255).permute(dims).cpu().numpy().astype(np.uint8)

        if array.shape[0] == 1:
            array = array.squeeze(0)
        if array.shape[-1] == 1:
            array = array.squeeze(-1)

        return array

    def __repr__(self):
        return self.__class__.__name__ + "()"
