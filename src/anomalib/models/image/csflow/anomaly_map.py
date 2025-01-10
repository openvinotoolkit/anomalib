"""Anomaly Map Generator for CS-Flow model.

This module provides functionality to generate anomaly maps from the CS-Flow model's
outputs. The generator can operate in two modes:

1. ``ALL`` - Combines anomaly scores from all scales (default)
2. ``MAX`` - Uses only the largest scale as mentioned in the paper

The anomaly maps are generated by computing the mean of squared z-scores across
channels and upsampling to the input dimensions.

Example:
    >>> import torch
    >>> generator = AnomalyMapGenerator(input_dims=(3, 256, 256))
    >>> z_dist = [torch.randn(2, 64, 32, 32) for _ in range(3)]
    >>> anomaly_map = generator(z_dist)
    >>> anomaly_map.shape
    torch.Size([2, 1, 256, 256])
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


class AnomalyMapMode(str, Enum):
    """Mode for generating anomaly maps.

    The mode determines how the anomaly scores from different scales are combined:

    - ``ALL``: Combines scores from all scales by multiplication
    - ``MAX``: Uses only the score from the largest scale
    """

    ALL = "all"
    MAX = "max"


class AnomalyMapGenerator(nn.Module):
    """Generate anomaly maps from CS-Flow model outputs.

    Args:
        input_dims (tuple[int, int, int]): Input dimensions in the format
            ``(channels, height, width)``.
        mode (AnomalyMapMode, optional): Mode for generating anomaly maps.
            Defaults to ``AnomalyMapMode.ALL``.

    Example:
        >>> generator = AnomalyMapGenerator((3, 256, 256))
        >>> z_dist = [torch.randn(1, 64, 32, 32) for _ in range(3)]
        >>> anomaly_map = generator(z_dist)
    """

    def __init__(self, input_dims: tuple[int, int, int], mode: AnomalyMapMode = AnomalyMapMode.ALL) -> None:
        super().__init__()
        self.mode = mode
        self.input_dims = input_dims

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Generate anomaly maps from z-distributions.

        Args:
            inputs (torch.Tensor): List of z-distributions from different scales,
                where each element has shape ``(batch_size, channels, height,
                width)``.

        Returns:
            torch.Tensor: Anomaly maps with shape ``(batch_size, 1, height,
                width)``, where height and width match the input dimensions.

        Example:
            >>> z_dist = [torch.randn(2, 64, 32, 32) for _ in range(3)]
            >>> generator = AnomalyMapGenerator((3, 256, 256))
            >>> maps = generator(z_dist)
            >>> maps.shape
            torch.Size([2, 1, 256, 256])
        """
        anomaly_map: torch.Tensor
        if self.mode == AnomalyMapMode.ALL:
            anomaly_map = torch.ones(inputs[0].shape[0], 1, *self.input_dims[1:]).to(inputs[0].device)
            for z_dist in inputs:
                mean_z = (z_dist**2).mean(dim=1, keepdim=True)
                anomaly_map *= F.interpolate(
                    mean_z,
                    size=self.input_dims[1:],
                    mode="bilinear",
                    align_corners=False,
                )
        else:
            mean_z = (inputs[0] ** 2).mean(dim=1, keepdim=True)
            anomaly_map = F.interpolate(
                mean_z,
                size=self.input_dims[1:],
                mode="bilinear",
                align_corners=False,
            )

        return anomaly_map
