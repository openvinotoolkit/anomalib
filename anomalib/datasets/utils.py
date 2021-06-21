"""
Dataset Utils
"""

from typing import List, Optional, Tuple

import numpy as np
import torch


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

        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def __call__(self, tensor: torch.Tensor) -> np.ndarray:
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

    def __call__(self, tensor: torch.Tensor, dims: Optional[Tuple[int, ...]] = None) -> np.ndarray:

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
