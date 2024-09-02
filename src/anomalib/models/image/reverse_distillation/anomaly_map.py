"""Compute Anomaly map."""

# Original Code
# Copyright (c) 2022 hq-deng
# https://github.com/hq-deng/RD4AD
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

import torch
from omegaconf import ListConfig
from torch import nn
from torch.nn import functional as F  # noqa: N812

from anomalib.models.components import GaussianBlur2d


class AnomalyMapGenerationMode(str, Enum):
    """Type of mode when generating anomaly imape."""

    ADD = "add"
    MULTIPLY = "multiply"


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap.

    Args:
        image_size (ListConfig, tuple): Size of original image used for upscaling the anomaly map.
        sigma (int): Standard deviation of the gaussian kernel used to smooth anomaly map.
            Defaults to ``4``.
        mode (AnomalyMapGenerationMode, optional): Operation used to generate anomaly map.
            Options are ``AnomalyMapGenerationMode.ADD`` and ``AnomalyMapGenerationMode.MULTIPLY``.
            Defaults to ``AnomalyMapGenerationMode.MULTIPLY``.

    Raises:
        ValueError: In case modes other than multiply and add are passed.
    """

    def __init__(
        self,
        image_size: ListConfig | tuple,
        sigma: int = 4,
        mode: AnomalyMapGenerationMode = AnomalyMapGenerationMode.MULTIPLY,
    ) -> None:
        super().__init__()
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)
        self.sigma = sigma
        self.kernel_size = 2 * int(4.0 * sigma + 0.5) + 1

        if mode not in {AnomalyMapGenerationMode.ADD, AnomalyMapGenerationMode.MULTIPLY}:
            msg = f"Found mode {mode}. Only multiply and add are supported."
            raise ValueError(msg)
        self.mode = mode

    def forward(self, student_features: list[torch.Tensor], teacher_features: list[torch.Tensor]) -> torch.Tensor:
        """Compute anomaly map given encoder and decoder features.

        Args:
            student_features (list[torch.Tensor]): List of encoder features
            teacher_features (list[torch.Tensor]): List of decoder features

        Returns:
            Tensor: Anomaly maps of length batch.
        """
        if self.mode == AnomalyMapGenerationMode.MULTIPLY:
            anomaly_map = torch.ones(
                [student_features[0].shape[0], 1, *self.image_size],
                device=student_features[0].device,
            )  # b c h w
        elif self.mode == AnomalyMapGenerationMode.ADD:
            anomaly_map = torch.zeros(
                [student_features[0].shape[0], 1, *self.image_size],
                device=student_features[0].device,
            )

        for student_feature, teacher_feature in zip(student_features, teacher_features, strict=True):
            distance_map = 1 - F.cosine_similarity(student_feature, teacher_feature)
            distance_map = torch.unsqueeze(distance_map, dim=1)
            distance_map = F.interpolate(distance_map, size=self.image_size, mode="bilinear", align_corners=True)
            if self.mode == AnomalyMapGenerationMode.MULTIPLY:
                anomaly_map *= distance_map
            elif self.mode == AnomalyMapGenerationMode.ADD:
                anomaly_map += distance_map

        gaussian_blur = GaussianBlur2d(
            kernel_size=(self.kernel_size, self.kernel_size),
            sigma=(self.sigma, self.sigma),
        ).to(student_features[0].device)

        return gaussian_blur(anomaly_map)
