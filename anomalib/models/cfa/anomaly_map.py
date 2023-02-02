"""Anomaly Map Generator for the CFA model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import ListConfig
from torch import Tensor, nn

from anomalib.models.components import GaussianBlur2d


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap."""

    def __init__(
        self,
        image_size: ListConfig | tuple,
        num_nearest_neighbors: int,
        sigma: int = 4,
    ) -> None:
        super().__init__()
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)
        self.num_nearest_neighbors = num_nearest_neighbors
        self.sigma = sigma

    def compute_score(self, distance: Tensor, scale: tuple[int, int]) -> Tensor:
        """Compute score based on the distance.

        Args:
            distance (Tensor): Distance tensor computed using target oriented
                features.
            scale (tuple[int, int]): Height and width of the largest feature
                map.

        Returns:
            Tensor: Score value.
        """
        distance = torch.sqrt(distance)
        distance = distance.topk(self.num_nearest_neighbors, largest=False).values
        distance = (F.softmin(distance, dim=-1)[:, :, 0]) * distance[:, :, 0]
        distance = distance.unsqueeze(-1)

        score = rearrange(distance, "b (h w) c -> b c h w", h=scale[0], w=scale[1])
        return score.detach()

    def compute_anomaly_map(self, score: Tensor) -> Tensor:
        """Compute anomaly map based on the score.

        Args:
            score (Tensor): Score tensor.

        Returns:
            Tensor: Anomaly map.
        """
        anomaly_map = score.mean(dim=1, keepdim=True)
        anomaly_map = F.interpolate(anomaly_map, size=self.image_size, mode="bilinear", align_corners=False)

        gaussian_blur = GaussianBlur2d(sigma=self.sigma).to(score.device)
        anomaly_map = gaussian_blur(anomaly_map)  # pylint: disable=not-callable
        return anomaly_map

    def forward(self, **kwargs) -> Tensor:
        """Return anomaly map.

        Raises:
            ``distance`` and ``scale`` keys are not found.

        Returns:
            Tensor: Anomaly heatmap.
        """
        if not ("distance" in kwargs and "scale" in kwargs):
            raise ValueError(f"Expected keys `distance` and `scale. Found {kwargs.keys()}")

        distance: Tensor = kwargs["distance"]
        scale: tuple[int, int] = kwargs["scale"]

        score = self.compute_score(distance=distance, scale=scale)
        anomaly_map = self.compute_anomaly_map(score)

        return anomaly_map
