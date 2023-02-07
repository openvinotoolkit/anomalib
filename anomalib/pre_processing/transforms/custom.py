"""Dataset Utils."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings

import numpy as np
import torch
from torch import Tensor


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
