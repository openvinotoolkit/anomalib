"""Anomaly Map Generator for the PatchCore model implementation."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.models.components import GaussianBlur2d


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap.

    Args:
        The anomaly map is upsampled to this dimension.
        sigma (int, optional): Standard deviation for Gaussian Kernel.
            Defaults to ``4``.
    """

    def __init__(
        self,
        sigma: int = 4,
    ) -> None:
        super().__init__()
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma), channels=1)

    def compute_anomaly_map(
        self,
        patch_scores: torch.Tensor,
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        """Pixel Level Anomaly Heatmap.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores
            image_size (tuple[int, int] | torch.Size, optional): Size of the input image.
                The anomaly map is upsampled to this dimension.
                Defaults to None.

        Returns:
            Tensor: Map of the pixel-level anomaly scores
        """
        if image_size is None:
            anomaly_map = patch_scores
        else:
            anomaly_map = F.interpolate(patch_scores, size=(image_size[0], image_size[1]))
        return self.blur(anomaly_map)

    def forward(
        self,
        patch_scores: torch.Tensor,
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        """Return anomaly_map and anomaly_score.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores
            image_size (tuple[int, int] | torch.Size, optional): Size of the input image.
                The anomaly map is upsampled to this dimension.
                Defaults to None.

        Example:
            >>> anomaly_map_generator = AnomalyMapGenerator()
            >>> map = anomaly_map_generator(patch_scores=patch_scores)

        Returns:
            Tensor: anomaly_map
        """
        return self.compute_anomaly_map(patch_scores, image_size)
