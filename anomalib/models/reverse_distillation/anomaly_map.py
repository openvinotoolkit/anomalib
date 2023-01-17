"""Compute Anomaly map."""

# Original Code
# Copyright (c) 2022 hq-deng
# https://github.com/hq-deng/RD4AD
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from omegaconf import ListConfig
from torch import Tensor, nn


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap.

    Args:
        image_size (ListConfig, tuple): Size of original image used for upscaling the anomaly map.
        sigma (int): Standard deviation of the gaussian kernel used to smooth anomaly map.
        mode (str, optional): Operation used to generate anomaly map. Options are `add` and `multiply`.
                Defaults to "multiply".

    Raises:
        ValueError: In case modes other than multiply and add are passed.
    """

    def __init__(self, image_size: ListConfig | tuple, sigma: int = 4, mode: str = "multiply") -> None:
        super().__init__()
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)
        self.sigma = sigma
        self.kernel_size = 2 * int(4.0 * sigma + 0.5) + 1

        if mode not in ("add", "multiply"):
            raise ValueError(f"Found mode {mode}. Only multiply and add are supported.")
        self.mode = mode

    def forward(self, student_features: list[Tensor], teacher_features: list[Tensor]) -> Tensor:
        """Computes anomaly map given encoder and decoder features.

        Args:
            student_features (list[Tensor]): List of encoder features
            teacher_features (list[Tensor]): List of decoder features

        Returns:
            Tensor: Anomaly maps of length batch.
        """
        if self.mode == "multiply":
            anomaly_map = torch.ones(
                [student_features[0].shape[0], 1, *self.image_size], device=student_features[0].device
            )  # b c h w
        elif self.mode == "add":
            anomaly_map = torch.zeros(
                [student_features[0].shape[0], 1, *self.image_size], device=student_features[0].device
            )

        for student_feature, teacher_feature in zip(student_features, teacher_features):
            distance_map = 1 - F.cosine_similarity(student_feature, teacher_feature)
            distance_map = torch.unsqueeze(distance_map, dim=1)
            distance_map = F.interpolate(distance_map, size=self.image_size, mode="bilinear", align_corners=True)
            if self.mode == "multiply":
                anomaly_map *= distance_map
            elif self.mode == "add":
                anomaly_map += distance_map

        anomaly_map = gaussian_blur2d(
            anomaly_map, kernel_size=(self.kernel_size, self.kernel_size), sigma=(self.sigma, self.sigma)
        )

        return anomaly_map
