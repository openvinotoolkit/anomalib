"""Dataset Utils."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings

import numpy as np
import torch
from albumentations.core.transforms_interface import ImageOnlyTransform
from torch import Tensor


class ToRGB(ImageOnlyTransform):
    """Convert BGR image to RGB.

    Args:
        always_apply (bool, optional): Always apply . Defaults to True.
        p (float, optional): Probability. Defaults to 1.0.

    Raises:
        TypeError:

    Examples:
        >>> import cv2
        >>> import numpy as np
        >>> from anomalib.data.transforms import ToRGB

        >>> bgr = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        >>> rgb = ToRGB()(image=bgr)["image"]
        >>> expected_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        >>> np.allclose(rgb, expected_rgb)
        True

        ToRGB also supports single channel images.
        >>> img = np.random.randint(0, 255, (1080, 1920), dtype=np.uint8)
        >>> rgb = ToRGB()(image=img)

        Single channel images are converted to 3-channel images, so each channel is equal.
        >>> (rgb[..., 0] == rgb[..., 1]).all()
        True
        >>> (rgb[..., 1] == rgb[..., 2]).all()
        True

    Returns:
        np.ndarray: RGB image converted from BGR.
    """

    def __init__(self, always_apply=True, p=1.0) -> None:
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Apply transformation.

        Args:
            img (np.ndarray): Image to transform.

        Raises:
            TypeError: When image is not 3-dim or last dimension is not equal to 3.

        Returns:
            np.ndarray: Transformed image.
        """
        del params  # params parameter is required for apply method, but not used.

        # Check if image is 1-channel and convert to 3-channel to apply transformation.
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)

        if len(img.shape) != 3 and img.shape[-1] != 3:
            raise TypeError("ToRGB transformation expects 3-dim images with the last dimension equal to 3.")

        return img[..., [2, 1, 0]]

    def get_transform_init_args_names(self):
        return ()


class Denormalize:
    """Denormalize Torch Tensor into np image format."""

    def __init__(self, mean: list[float] | None = None, std: list[float] | None = None) -> None:
        """Denormalize Torch Tensor into np image format.

        Args:
            mean: Mean
            std: Standard deviation.
        """
        warnings.warn("Denormalize is no longer used and will be deprecated in v0.4.0")
        # If no mean and std provided, assign ImageNet values.
        if mean is None:
            mean = [0.485, 0.456, 0.406]

        if std is None:
            std = [0.229, 0.224, 0.225]

        self.mean = Tensor(mean)
        self.std = Tensor(std)

    def __call__(self, tensor: Tensor) -> np.ndarray:
        """Denormalize the input.

        Args:
            tensor (Tensor): Input tensor image (C, H, W)

        Returns:
            Denormalized numpy array (H, W, C).
        """
        if tensor.dim() == 4:
            if tensor.size(0):
                tensor = tensor.squeeze(0)
            else:
                raise ValueError(f"Tensor has batch size of {tensor.size(0)}. Only single batch is supported.")

        denormalized_per_channel = [(tnsr * std) + mean for tnsr, mean, std in zip(tensor, self.mean, self.std)]
        denormalized_tensor = torch.stack(denormalized_per_channel)

        denormalized_array = (denormalized_tensor * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return denormalized_array

    def __repr__(self) -> str:
        """Representational string."""
        return self.__class__.__name__ + "()"


class ToNumpy:
    """Convert Tensor into Numpy Array."""

    def __call__(self, tensor: Tensor, dims: tuple[int, ...] | None = None) -> np.ndarray:
        """Convert Tensor into Numpy Array.

        Args:
           tensor (Tensor): Tensor to convert. Input tensor in range 0-1.
           dims (tuple[int, ...] | None, optional): Convert dimensions from torch to numpy format.
                Tuple corresponding to axis permutation from torch tensor to numpy array. Defaults to None.

        Returns:
            Converted numpy ndarray.
        """
        # Default support is (C, H, W) or (N, C, H, W)
        if dims is None:
            dims = (0, 2, 3, 1) if len(tensor.shape) == 4 else (1, 2, 0)

        array = (tensor * 255).permute(dims).cpu().numpy().astype(np.uint8)

        if array.shape[0] == 1:
            array = array.squeeze(0)
        if array.shape[-1] == 1:
            array = array.squeeze(-1)

        return array

    def __repr__(self) -> str:
        """Representational string."""
        return self.__class__.__name__ + "()"
