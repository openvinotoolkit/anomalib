"""Anomaly Map Generator for the CFA model implementation.

This module provides functionality to generate anomaly heatmaps from distance
features computed by the CFA model.

Example:
    >>> import torch
    >>> from anomalib.models.image.cfa.anomaly_map import AnomalyMapGenerator
    >>> # Initialize generator
    >>> generator = AnomalyMapGenerator(num_nearest_neighbors=3)
    >>> # Generate anomaly map
    >>> distance = torch.randn(1, 1024, 1)  # batch x pixels x 1
    >>> scale = (32, 32)  # height x width
    >>> anomaly_map = generator(distance=distance, scale=scale)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.models.components import GaussianBlur2d


class AnomalyMapGenerator(nn.Module):
    """Generate anomaly heatmaps from distance features.

    The generator computes anomaly scores based on k-nearest neighbor distances
    and applies Gaussian smoothing to produce the final heatmap.

    Args:
        num_nearest_neighbors (int): Number of nearest neighbors to consider
            when computing anomaly scores.
        sigma (int, optional): Standard deviation for Gaussian smoothing.
            Defaults to ``4``.

    Example:
        >>> import torch
        >>> generator = AnomalyMapGenerator(num_nearest_neighbors=3)
        >>> distance = torch.randn(1, 1024, 1)  # batch x pixels x 1
        >>> scale = (32, 32)  # height x width
        >>> anomaly_map = generator(distance=distance, scale=scale)
    """

    def __init__(
        self,
        num_nearest_neighbors: int,
        sigma: int = 4,
    ) -> None:
        super().__init__()
        self.num_nearest_neighbors = num_nearest_neighbors
        self.sigma = sigma

    def compute_score(self, distance: torch.Tensor, scale: tuple[int, int]) -> torch.Tensor:
        """Compute anomaly scores from distance features.

        Args:
            distance (torch.Tensor): Distance tensor of shape
                ``(batch_size, num_pixels, 1)``.
            scale (tuple[int, int]): Height and width of the feature map used
                to reshape the scores.

        Returns:
            torch.Tensor: Anomaly scores of shape
                ``(batch_size, 1, height, width)``.
        """
        distance = torch.sqrt(distance)
        distance = distance.topk(self.num_nearest_neighbors, largest=False).values  # noqa: PD011
        distance = (F.softmin(distance, dim=-1)[:, :, 0]) * distance[:, :, 0]
        distance = distance.unsqueeze(-1)

        score = rearrange(distance, "b (h w) c -> b c h w", h=scale[0], w=scale[1])
        return score.detach()

    def compute_anomaly_map(
        self,
        score: torch.Tensor,
        image_size: tuple[int, int] | torch.Size | None = None,
    ) -> torch.Tensor:
        """Generate smoothed anomaly map from scores.

        Args:
            score (torch.Tensor): Anomaly scores of shape
                ``(batch_size, 1, height, width)``.
            image_size (tuple[int, int] | torch.Size | None, optional): Target
                size for upsampling the anomaly map. Defaults to ``None``.

        Returns:
            torch.Tensor: Smoothed anomaly map of shape
                ``(batch_size, 1, height, width)``.
        """
        anomaly_map = score.mean(dim=1, keepdim=True)
        if image_size is not None:
            anomaly_map = F.interpolate(anomaly_map, size=image_size, mode="bilinear", align_corners=False)

        gaussian_blur = GaussianBlur2d(sigma=self.sigma).to(score.device)
        return gaussian_blur(anomaly_map)  # pylint: disable=not-callable

    def forward(self, **kwargs) -> torch.Tensor:
        """Generate anomaly map from input features.

        The method expects ``distance`` and ``scale`` as required inputs, with
        optional ``image_size`` for upsampling.

        Args:
            **kwargs: Keyword arguments containing:
                - distance (torch.Tensor): Distance features
                - scale (tuple[int, int]): Feature map scale
                - image_size (tuple[int, int] | torch.Size, optional):
                    Target size for upsampling

        Raises:
            ValueError: If required arguments are missing.

        Returns:
            torch.Tensor: Anomaly heatmap of shape
                ``(batch_size, 1, height, width)``.
        """
        if not ("distance" in kwargs and "scale" in kwargs):
            msg = f"Expected keys `distance` and `scale`. Found {kwargs.keys()}"
            raise ValueError(msg)

        distance: torch.Tensor = kwargs["distance"]
        scale: tuple[int, int] = kwargs["scale"]
        image_size: tuple[int, int] | torch.Size | None = kwargs.get("image_size", None)

        score = self.compute_score(distance=distance, scale=scale)
        return self.compute_anomaly_map(score, image_size=image_size)
