"""Anomaly Map Generator for the PatchCore model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union

import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from torch import nn

from anomalib.models.components import GaussianBlur2d


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap."""

    def __init__(
        self,
        input_size: Union[ListConfig, Tuple],
        sigma: int = 4,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma), channels=1)

    def compute_anomaly_map(self, patch_scores: torch.Tensor, feature_map_shape: torch.Size) -> torch.Tensor:
        """Pixel Level Anomaly Heatmap.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores
            feature_map_shape (torch.Size): 2-D feature map shape (width, height)

        Returns:
            torch.Tensor: Map of the pixel-level anomaly scores
        """
        width, height = feature_map_shape
        batch_size = len(patch_scores) // (width * height)

        anomaly_map = patch_scores[:, 0].reshape((batch_size, 1, width, height))
        anomaly_map = F.interpolate(anomaly_map, size=(self.input_size[0], self.input_size[1]))

        anomaly_map = self.blur(anomaly_map)

        return anomaly_map

    @staticmethod
    def compute_anomaly_score(patch_scores: torch.Tensor) -> torch.Tensor:
        """Compute Image-Level Anomaly Score.

        Args:
            patch_scores (torch.Tensor): Patch-level anomaly scores
        Returns:
            torch.Tensor: Image-level anomaly scores
        """
        max_scores = torch.argmax(patch_scores[:, 0])
        confidence = torch.index_select(patch_scores, 0, max_scores)
        weights = 1 - (torch.max(torch.exp(confidence)) / torch.sum(torch.exp(confidence)))
        score = weights * torch.max(patch_scores[:, 0])
        return score

    def forward(self, **kwargs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns anomaly_map and anomaly_score.

        Expects `patch_scores` keyword to be passed explicitly
        Expects `feature_map_shape` keyword to be passed explicitly

        Example
        >>> anomaly_map_generator = AnomalyMapGenerator(input_size=input_size)
        >>> map, score = anomaly_map_generator(patch_scores=numpy_array, feature_map_shape=feature_map_shape)

        Raises:
            ValueError: If `patch_scores` key is not found

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: anomaly_map, anomaly_score
        """

        if "patch_scores" not in kwargs:
            raise ValueError(f"Expected key `patch_scores`. Found {kwargs.keys()}")

        if "feature_map_shape" not in kwargs:
            raise ValueError(f"Expected key `feature_map_shape`. Found {kwargs.keys()}")

        patch_scores = kwargs["patch_scores"]
        feature_map_shape = kwargs["feature_map_shape"]

        anomaly_map = self.compute_anomaly_map(patch_scores, feature_map_shape)
        anomaly_score = self.compute_anomaly_score(patch_scores)
        return anomaly_map, anomaly_score
