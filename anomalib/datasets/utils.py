"""
Dataset Utils
"""
import warnings
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from omegaconf import ListConfig
from torch import Tensor
from torch.nn import functional as F


class StrideSizeError(Exception):
    """
    StrideSizeError to raise exception when stride
        size is greater than the tile size.

    Args:
        Exception ([type]): [description]
    """


def pad_image(image: Tensor, input_size: Tuple, tile_size: Tuple, stride: Tuple) -> Tensor:
    """
    This function pads image when the images size is not divisible by tile size and stride.

    Args:
        image (Tensor): Input image
        input_size (Tuple): Size of the input height and width
        tile_size (Tuple): Tile size - height and width
        stride (Tuple): Stride - height and width

    Examples:

        >>> image = torch.rand(1, 3, 512, 512)
        >>> image = pad_image(image, input_size=(512, 512), tile_size=(256, 256), stride=(128, 128))
        >>> image.shape
        torch.Size([1, 3, 512, 512])

        >>> image = torch.rand(1, 3, 512, 512)
        >>> image = pad_image(image, input_size=(512, 512), tile_size=(211, 200), stride=(99, 27))
        >>> image.shape
        torch.Size([1, 3, 686, 706])

    Returns:
        Tensor: [description]
    """

    def __pad_edge(edge_size: int, tile_size: int, stride: int) -> int:
        """
        __pad_edge [summary]

        Args:
            edge_size (int): [description]
            tile_size (int): [description]
            stride (int): [description]

        Returns:
            int: [description]
        """
        if edge_size % tile_size == 0 and edge_size % stride == 0:
            pad = 0
        else:
            pad = ((edge_size // stride * stride) + tile_size) - edge_size

        return pad

    pad_h = __pad_edge(input_size[0], tile_size[0], stride[0])
    pad_w = __pad_edge(input_size[1], tile_size[1], stride[1])

    image = F.pad(image, [0, pad_w, 0, pad_h])
    # image = F.pad(image, [0, pad_h, 0, pad_w])

    return image


def interpolate_image(image: Tensor, input_size: Tuple, tile_size: Tuple, stride: Tuple) -> Tensor:
    """
    This function interpolates image when the images size is not divisible by tile size and stride.

    Args:
        image (Tensor): Input image
        input_size (Tuple): Size of the input height and width
        tile_size (Tuple): Tile size - height and width
        stride (Tuple): Stride - height and width

    Examples:
        >>> image = torch.rand(1, 3, 512, 512)
        >>> image = interpolate_image(image, input_size=(512, 512), tile_size=(256, 256), stride=(128, 128))
        >>> image.shape
        torch.Size([1, 3, 512, 512])

        >>> image = torch.rand(1, 3, 512, 512)
        >>> image = interpolate_image(image, input_size=(512, 512), tile_size=(211, 200), stride=(99, 27))
        >>> image.shape
        torch.Size([1, 3, 706, 686])

    Returns:
        Tensor: [description]
    """

    def resize_edge(edge_size: int, tile_size: int, stride: int) -> int:
        """
        Resizes edges using tile size and stride via interpolation.

        Args:
            edge_size (int): Edge Size (Height or Width)
            tile_size (int): Tile Size (Height or Width)
            stride (int): Stride Size (Height or Width)

        Returns:
            int: Resized edge size.
        """

        if edge_size % tile_size == 0 and edge_size % stride == 0:
            resized_edge = edge_size
        else:
            resized_edge = edge_size // stride * stride + tile_size

        return resized_edge

    # Resize the image if tile size is not divisable.
    resized_h = resize_edge(input_size[0], tile_size[0], stride[0])
    resized_w = resize_edge(input_size[1], tile_size[1], stride[1])
    image = F.interpolate(input=image, size=(resized_h, resized_w))
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

    # def __init__(self, tile_size: int, stride: int) -> None:
    def __init__(
        self,
        tile_size: Union[int, Sequence],
        stride: Union[int, Sequence],
        mode: str = "interpolation",
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

        if self.mode == "padding":
            warnings.warn(
                "Padding is not stable in this version. "
                "If you want more stable results for edge cases, consider using interpolation."
            )

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
            raise ValueError(
                    f"Unknown type {type(parameter)} for tile or stride size. Could be int or Sequence type."
            )

        if len(output) != 2:
            raise ValueError(f"Length of the size type must be 2 for height and width. Got {len(output)} instead.")

        # if isinstance(parameter, tuple):
        #     output = parameter
        # else:
        #     if isinstance(parameter, int):
        #         output = (parameter,) * 2
        #     elif isinstance(parameter, ListConfig):
        #         output = tuple(parameter)
        #     else:
        #         raise ValueError(
        #             f"Unknown type {type(parameter)} for tile or stride size. Could be int, tuple or ListConfig"
        #         )
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

        # Resize image if needed
        resize = pad_image if self.mode == "padding" else interpolate_image
        image = resize(
            image=image,
            input_size=(self.input_h, self.input_w),
            tile_size=(self.tile_size_h, self.tile_size_w),
            stride=(self.stride_h, self.stride_w),
        )
        self.resized_h, self.resized_w = image.shape[2:]

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

        if self.mode == "padding":
            image = image[:, :, : self.input_h, : self.input_w]
        else:
            image = F.interpolate(input=image, size=(self.input_h, self.input_w))

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
