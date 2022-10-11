"""Anomaly Map Generator for the PaDiM model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Tuple, Union

import einops
import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from torch import Tensor, nn

from anomalib.models.components import GaussianBlur2d


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap.

    Args:
        image_size (Union[ListConfig, Tuple]): Size of the input image. The anomaly map is upsampled to this dimension.
        sigma (int, optional): Standard deviation for Gaussian Kernel. Defaults to 4.
    """

    def __init__(
        self, image_size: Union[ListConfig, Tuple], layer_size: int, num_nearest_neighbors: int = 3, sigma: int = 4
    ) -> None:
        super().__init__()
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)
        # TODO: Currently this is int. Ideally it should be hxw
        self.layer_size = layer_size
        self.num_nearest_neighbors = num_nearest_neighbors
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.gaussian_blur = GaussianBlur2d(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma), channels=1)

    def compute_score(self, distance: Tensor) -> Tensor:
        distance = torch.sqrt(distance)

        n_neighbors = self.num_nearest_neighbors
        distance = distance.topk(n_neighbors, largest=False).values

        distance = (F.softmin(distance, dim=-1)[:, :, 0]) * distance[:, :, 0]
        distance = distance.unsqueeze(-1)

        score = einops.rearrange(distance, "b (h w) c -> b c h w", h=self.layer_size)
        return score.detach()

    def up_sample(self, distance: Tensor) -> Tensor:
        """Up sample anomaly score to match the input image size.

        Args:
            distance (Tensor): Anomaly score computed via the mahalanobis distance.

        Returns:
            Resized distance matrix matching the input image size
        """

        score_map = F.interpolate(
            distance,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )
        return score_map

    def forward(self, **kwargs: Dict[str, Tensor]) -> Tensor:
        """Returns anomaly_map.

        Expects ``distance`` keyword to be passed explicitly.

        Example:
        >>> anomaly_map_generator = AnomalyMapGenerator(image_size=input_size)
        >>> output = anomaly_map_generator(distance=distance)

        Raises:
            ValueError: ``distance`` key is not found.

        Returns:
            Tensor: anomaly map
        """

        if not ("distance" in kwargs):
            raise ValueError(f"Expected keys ``distance``. Found {kwargs.keys()}")

        distance: Tensor = kwargs["distance"]

        anomaly_score = self.compute_score(distance=distance)

        anomaly_map = torch.mean(anomaly_score, dim=1, keepdim=True)
        anomaly_map = self.up_sample(anomaly_map)
        anomaly_map = self.gaussian_blur(anomaly_map)

        return anomaly_map
