"""
Dataset Utils
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision
from torch import Size, Tensor, nn


class Patchify:
    """
    Patchify Image.
    """

    def __init__(self, patch_size: int = 64, stride: int = 64, padding: int = 0, dilation: int = 1):
        self.patch_size = patch_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

        self.batch_size: int
        self.image_size: Size

    def split_image(self, image: Tensor) -> Tensor:
        """
        Split an image into patches.

        Args:
            image: Input image

        Returns:
            Patches from the original input image.

        """

        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        self.image_size = image.shape[2:]
        num_channels = image.shape[1]

        # Image patches of size NxF**2xP, where F: patch size, P: # of patches.
        image_patches = nn.Unfold(self.patch_size, self.dilation, self.padding, self.stride)(image)

        # Permute dims to have the following dim: NxPxF**2
        image_patches = image_patches.permute(0, 2, 1)

        # converted tensor into NxPxHXW, Reshape patches into PxCxFxF
        image_patches = image_patches.reshape(image_patches.shape[1], num_channels, self.patch_size, self.patch_size)

        return image_patches

    def split_batch(self, batch: Tensor) -> Tensor:
        """
        Split Image Batch into Patches

        Args:
            batch (Tensor): Batch of images with NxCxHxW dims.

        Returns:
            Tensor: Patches of batch of images.
        """

        self.batch_size = batch.shape[0]
        self.image_size = batch.shape[2:]

        batch_patch_list: List[Tensor] = [self.split_image(image) for image in batch]
        batch_patches: Tensor = torch.cat(batch_patch_list, dim=0)

        return batch_patches

    def merge_image_patches(
        self,
        patches: Tensor,
        padding: int = 0,
        normalize: bool = True,
        pixel_range: Optional[tuple] = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> Tensor:
        """
        Merge the patches to form the original image.
        Args:
            patches: Patches to merge (stitch)
            padding: Number of pixels to skip when stitching the patches.
            normalize: Normalize the output image.
            pixel_range: Pixel range of the output image.
            scale_each: Scale each patch before merging.
            pad_value: Pixel value of the pads between patches.

        Returns:
            Output image by merging (stitching) the patches.

        """

        _, img_width = self.image_size
        num_rows = img_width // self.patch_size

        grid = torchvision.utils.make_grid(patches, num_rows, padding, normalize, pixel_range, scale_each, pad_value)

        return grid

    def merge_batch_patches(
        self,
        patches: Tensor,
        padding: int = 0,
        normalize: bool = True,
        pixel_range: Optional[tuple] = None,
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> Tensor:
        """
        Merge the patches to form the original batch.
        Args:
            patches: Patches to merge (stitch)
            padding: Number of pixels to skip when stitching the patches.
            normalize: Normalize the output image.
            pixel_range: Pixel range of the output image.
            scale_each: Scale each patch before merging.
            pad_value: Pixel value of the pads between patches.

        Returns:
            Output image by merging (stitching) the patches.

        """

        batch_list: List[Tensor] = []
        batch_patches = torch.chunk(input=patches, chunks=self.batch_size, dim=0)

        for image_patches in batch_patches:
            image = self.merge_image_patches(image_patches, padding, normalize, pixel_range, scale_each, pad_value)
            batch_list.append(image)

        batch = torch.cat(batch_list, dim=0)

        return batch

    def __call__(self, batch: Tensor) -> Tensor:
        return self.split_batch(batch)


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
