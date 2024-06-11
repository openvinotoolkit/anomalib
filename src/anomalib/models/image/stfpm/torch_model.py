"""PyTorch model for the STFPM model implementation."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import nn

from anomalib.models.components import TimmFeatureExtractor

from .anomaly_map import AnomalyMapGenerator

if TYPE_CHECKING:
    from anomalib.data.utils.tiler import Tiler


class STFPMModel(nn.Module):
    """STFPM: Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection.

    Args:
        layers (list[str]): Layers used for feature extraction.
        backbone (str, optional): Pre-trained model backbone.
            Defaults to ``resnet18``.
    """

    def __init__(
        self,
        layers: Sequence[str],
        backbone: str = "resnet18",
    ) -> None:
        super().__init__()
        self.tiler: Tiler | None = None

        self.backbone = backbone
        self.teacher_model = TimmFeatureExtractor(backbone=self.backbone, pre_trained=True, layers=layers).eval()
        self.student_model = TimmFeatureExtractor(
            backbone=self.backbone,
            pre_trained=False,
            layers=layers,
            requires_grad=True,
        )

        # teacher model is fixed
        for parameters in self.teacher_model.parameters():
            parameters.requires_grad = False

        self.anomaly_map_generator = AnomalyMapGenerator()

    def forward(self, images: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor]]:
        """Forward-pass images into the network.

        During the training mode the model extracts the features from the teacher and student networks.
        During the evaluation mode, it returns the predicted anomaly map.

        Args:
          images (torch.Tensor): Batch of images.

        Returns:
          Teacher and student features when in training mode, otherwise the predicted anomaly maps.
        """
        output_size = images.shape[-2:]
        if self.tiler:
            images = self.tiler.tile(images)
        teacher_features: dict[str, torch.Tensor] = self.teacher_model(images)
        student_features: dict[str, torch.Tensor] = self.student_model(images)

        if self.tiler:
            for layer, data in teacher_features.items():
                teacher_features[layer] = self.tiler.untile(data)
            for layer, data in student_features.items():
                student_features[layer] = self.tiler.untile(data)

        if self.training:
            output = teacher_features, student_features
        else:
            output = self.anomaly_map_generator(
                teacher_features=teacher_features,
                student_features=student_features,
                image_size=output_size,
            )

        return output
