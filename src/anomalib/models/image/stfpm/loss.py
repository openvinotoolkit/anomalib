"""Loss function for the STFPM Model Implementation."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


class STFPMLoss(nn.Module):
    """Feature Pyramid Loss This class implmenents the feature pyramid loss function proposed in STFPM paper.

    Example:
        >>> from anomalib.models.components.feature_extractors import TimmFeatureExtractor
        >>> from anomalib.models.stfpm.loss import STFPMLoss
        >>> from torchvision.models import resnet18

        >>> layers = ['layer1', 'layer2', 'layer3']
        >>> teacher_model = TimmFeatureExtractor(model=resnet18(pretrained=True), layers=layers)
        >>> student_model = TimmFeatureExtractor(model=resnet18(pretrained=False), layers=layers)
        >>> loss = Loss()

        >>> inp = torch.rand((4, 3, 256, 256))
        >>> teacher_features = teacher_model(inp)
        >>> student_features = student_model(inp)
        >>> loss(student_features, teacher_features)
            tensor(51.2015, grad_fn=<SumBackward0>)
    """

    def __init__(self) -> None:
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def compute_layer_loss(self, teacher_feats: torch.Tensor, student_feats: torch.Tensor) -> torch.Tensor:
        """Compute layer loss based on Equation (1) in Section 3.2 of the paper.

        Args:
          teacher_feats (torch.Tensor): Teacher features
          student_feats (torch.Tensor): Student features

        Returns:
          L2 distance between teacher and student features.
        """
        height, width = teacher_feats.shape[2:]

        norm_teacher_features = F.normalize(teacher_feats)
        norm_student_features = F.normalize(student_feats)
        return (0.5 / (width * height)) * self.mse_loss(norm_teacher_features, norm_student_features)

    def forward(
        self,
        teacher_features: dict[str, torch.Tensor],
        student_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the overall loss via the weighted average of the layer losses computed by the cosine similarity.

        Args:
          teacher_features (dict[str, torch.Tensor]): Teacher features
          student_features (dict[str, torch.Tensor]): Student features

        Returns:
          Total loss, which is the weighted average of the layer losses.
        """
        layer_losses: list[torch.Tensor] = []
        for layer in teacher_features:
            loss = self.compute_layer_loss(teacher_features[layer], student_features[layer])
            layer_losses.append(loss)

        return torch.stack(layer_losses).sum()
