"""Loss function for Student-Teacher Feature Pyramid Matching model.

This module implements the loss function used to train the STFPM model for anomaly
detection as described in `Wang et al. (2021) <https://arxiv.org/abs/2103.04257>`_.

The loss function:
1. Takes feature maps from teacher and student networks as input
2. Normalizes the features using L2 normalization
3. Computes MSE loss between normalized features
4. Scales the loss by spatial dimensions of feature maps

Example:
    >>> from anomalib.models.components import TimmFeatureExtractor
    >>> from anomalib.models.image.stfpm.loss import STFPMLoss
    >>> from torchvision.models import resnet18
    >>> layers = ["layer1", "layer2", "layer3"]
    >>> teacher_model = TimmFeatureExtractor(
    ...     model=resnet18(pretrained=True),
    ...     layers=layers
    ... )
    >>> student_model = TimmFeatureExtractor(
    ...     model=resnet18(pretrained=False),
    ...     layers=layers
    ... )
    >>> criterion = STFPMLoss()
    >>> features = torch.randn(4, 3, 256, 256)
    >>> teacher_features = teacher_model(features)
    >>> student_features = student_model(features)
    >>> loss = criterion(student_features, teacher_features)

See Also:
    - :class:`STFPMLoss`: Main loss class implementation
    - :class:`Stfpm`: Lightning implementation of the full model
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


class STFPMLoss(nn.Module):
    """Loss function for Student-Teacher Feature Pyramid Matching model.

    This class implements the feature pyramid loss function proposed in the STFPM
    paper. The loss measures the discrepancy between feature representations from
    a pre-trained teacher network and a student network that learns to match them.

    The loss computation involves:
    1. Normalizing teacher and student features using L2 normalization
    2. Computing MSE loss between normalized features
    3. Scaling the loss by spatial dimensions of feature maps
    4. Summing losses across all feature layers

    Example:
        >>> from anomalib.models.components import TimmFeatureExtractor
        >>> from anomalib.models.image.stfpm.loss import STFPMLoss
        >>> from torchvision.models import resnet18
        >>> layers = ["layer1", "layer2", "layer3"]
        >>> teacher_model = TimmFeatureExtractor(
        ...     model=resnet18(pretrained=True),
        ...     layers=layers
        ... )
        >>> student_model = TimmFeatureExtractor(
        ...     model=resnet18(pretrained=False),
        ...     layers=layers
        ... )
        >>> criterion = STFPMLoss()
        >>> features = torch.randn(4, 3, 256, 256)
        >>> teacher_features = teacher_model(features)
        >>> student_features = student_model(features)
        >>> loss = criterion(student_features, teacher_features)
        >>> loss
        tensor(51.2015, grad_fn=<SumBackward0>)

    See Also:
        - :class:`Stfpm`: Lightning implementation of the full model
        - :class:`STFPMModel`: PyTorch implementation of the model architecture
    """

    def __init__(self) -> None:
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction="sum")

    def compute_layer_loss(self, teacher_feats: torch.Tensor, student_feats: torch.Tensor) -> torch.Tensor:
        """Compute loss between teacher and student features for a single layer.

        This implements the loss computation based on Equation (1) in Section 3.2
        of the paper. The loss is computed as:
        1. L2 normalize teacher and student features
        2. Compute MSE loss between normalized features
        3. Scale loss by spatial dimensions (height * width)

        Args:
            teacher_feats (torch.Tensor): Features from teacher network with shape
                ``(B, C, H, W)``
            student_feats (torch.Tensor): Features from student network with shape
                ``(B, C, H, W)``

        Returns:
            torch.Tensor: Scalar loss value for the layer
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
        """Compute total loss across all feature layers.

        The total loss is computed as the sum of individual layer losses. Each
        layer loss measures the discrepancy between teacher and student features
        at that layer.

        Args:
            teacher_features (dict[str, torch.Tensor]): Dictionary mapping layer
                names to teacher feature tensors
            student_features (dict[str, torch.Tensor]): Dictionary mapping layer
                names to student feature tensors

        Returns:
            torch.Tensor: Total loss summed across all layers
        """
        layer_losses: list[torch.Tensor] = []
        for layer in teacher_features:
            loss = self.compute_layer_loss(teacher_features[layer], student_features[layer])
            layer_losses.append(loss)

        return torch.stack(layer_losses).sum()
