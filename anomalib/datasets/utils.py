from typing import List, Optional, Tuple

import numpy as np
import torch


class Denormalize(object):
    """ """
    def __init__(self, mean: Optional[List[float]] = None, std: Optional[List[float]] = None):
        # If no mean and std provided, assign ImageNet values.
        self.mean = mean if mean is not None else torch.Tensor([0.485, 0.456, 0.406])
        self.std = std if std is not None else torch.Tensor([0.229, 0.224, 0.225])

    def __call__(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Unnormalize the input
        :param tensor: Input tensor image (C, H, W)
        :return: Unnormalized numpy array (H, W, C).
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)

        array = (tensor * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return array

    def __repr__(self):
        return self.__class__.__name__ + "()"


class ToNumpy:
    """ """
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
