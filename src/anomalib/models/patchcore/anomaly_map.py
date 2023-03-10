"""Anomaly Map Generator for the PatchCore model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch.nn.functional as F
from omegaconf import ListConfig
from torch import Tensor, nn

from anomalib.models.components import GaussianBlur2d


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap."""

    def __init__(
        self,
        input_size: ListConfig | tuple,
        sigma: int = 4,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma), channels=1)

    def compute_anomaly_map(self, patch_scores: Tensor) -> Tensor:
        """Pixel Level Anomaly Heatmap.

        Args:
            patch_scores (Tensor): Patch-level anomaly scores

        Returns:
            Tensor: Map of the pixel-level anomaly scores
        """
        anomaly_map = F.interpolate(patch_scores, size=(self.input_size[0], self.input_size[1]))
        anomaly_map = self.blur(anomaly_map)

        return anomaly_map

    def forward(self, patch_scores: Tensor) -> Tensor:
        """Returns anomaly_map and anomaly_score.

        Args:
            patch_scores (Tensor): Patch-level anomaly scores

        Example
        >>> anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)
        >>> map = anomaly_map_generator(patch_scores=patch_scores)

        Returns:
            Tensor: anomaly_map
        """
        anomaly_map = self.compute_anomaly_map(patch_scores)
        return anomaly_map
