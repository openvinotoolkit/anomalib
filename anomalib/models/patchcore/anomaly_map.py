"""Anomaly Map Generator for the PatchCore model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, List

import torch.nn.functional as F
from omegaconf import ListConfig
from torch import Tensor, nn
from torch.jit import script_if_tracing
from torch.nn.functional import conv2d
import torch
from torchvision.transforms.functional_tensor import (
    _assert_image_tensor,
    _get_gaussian_kernel2d,
    _cast_squeeze_in,
    _cast_squeeze_out,
)
from kornia.filters import gaussian_blur2d


def compute_anomaly_map(patch_scores: torch.Tensor, input_size: Tuple[int, int], sigma: float) -> Tensor:
    """Pixel Level Anomaly Heatmap.

    Args:
        patch_scores: Patch-level anomaly scores
        input_size: 2-D input size
        sigma: Sigma value for the Gaussian kernel

    Returns:
        torch.Tensor: Map of the pixel-level anomaly scores
    """
    anomaly_map = F.interpolate(patch_scores, size=(input_size[0], input_size[1]))

    kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
    anomaly_map = gaussian_blur2d(anomaly_map, (kernel_size, kernel_size), sigma=(sigma, sigma))
    return anomaly_map


@script_if_tracing
def _forward(patch_scores: torch.Tensor, input_size: Tuple[int, int], sigma: float) -> Tensor:
    anomaly_map = compute_anomaly_map(patch_scores, input_size, sigma)
    return anomaly_map


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap."""

    def __init__(
        self,
        input_size: Tuple[int, int],
        sigma: int = 4,
    ) -> None:
        super(AnomalyMapGenerator, self).__init__()
        self.input_size = input_size
        self.sigma = sigma

    def forward(self, patch_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns anomaly_map and anomaly_score.

        Args:
            patch_scores (Tensor): Patch-level anomaly scores

        Example
        >>> anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)
        >>> map = anomaly_map_generator(patch_scores=patch_scores)

        Returns:
            Tensor: anomaly_map
        """
        return _forward(patch_scores, self.input_size, self.sigma)
