"""Loss function for Reverse Distillation model.

This module implements the loss function used to train the Reverse Distillation model
for anomaly detection. The loss is based on cosine similarity between encoder and
decoder features.

The loss function:
1. Takes encoder and decoder feature maps as input
2. Flattens the spatial dimensions of each feature map
3. Computes cosine similarity between corresponding encoder-decoder pairs
4. Averages the similarities across spatial dimensions and feature pairs

Example:
    >>> import torch
    >>> from anomalib.models.image.reverse_distillation.loss import (
    ...     ReverseDistillationLoss
    ... )
    >>> criterion = ReverseDistillationLoss()
    >>> encoder_features = [torch.randn(2, 64, 32, 32)]
    >>> decoder_features = [torch.randn(2, 64, 32, 32)]
    >>> loss = criterion(encoder_features, decoder_features)

See Also:
    - :class:`ReverseDistillationLoss`: Main loss class implementation
    - :class:`ReverseDistillation`: Lightning implementation of the full model
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn


class ReverseDistillationLoss(nn.Module):
    """Loss function for Reverse Distillation model.

    This class implements the cosine similarity loss used to train the Reverse
    Distillation model. The loss measures the dissimilarity between encoder and
    decoder feature maps.

    The loss computation involves:
    1. Flattening the spatial dimensions of encoder and decoder feature maps
    2. Computing cosine similarity between corresponding encoder-decoder pairs
    3. Subtracting similarities from 1 to get a dissimilarity measure
    4. Taking mean across spatial dimensions and feature pairs

    Example:
        >>> import torch
        >>> from anomalib.models.image.reverse_distillation.loss import (
        ...     ReverseDistillationLoss
        ... )
        >>> criterion = ReverseDistillationLoss()
        >>> encoder_features = [torch.randn(2, 64, 32, 32)]
        >>> decoder_features = [torch.randn(2, 64, 32, 32)]
        >>> loss = criterion(encoder_features, decoder_features)

    References:
        - Official Implementation:
          https://github.com/hq-deng/RD4AD/blob/main/main.py
        - Implementation Details:
          https://github.com/hq-deng/RD4AD/issues/22
    """

    @staticmethod
    def forward(encoder_features: list[torch.Tensor], decoder_features: list[torch.Tensor]) -> torch.Tensor:
        """Compute cosine similarity loss between encoder and decoder features.

        Args:
            encoder_features (list[torch.Tensor]): List of feature tensors from the
                encoder network. Each tensor has shape ``(B, C, H, W)`` where B is
                batch size, C is channels, H and W are spatial dimensions.
            decoder_features (list[torch.Tensor]): List of feature tensors from the
                decoder network. Must match encoder features in length and shapes.

        Returns:
            torch.Tensor: Scalar loss value computed as mean of (1 - cosine
                similarity) across all feature pairs.
        """
        cos_loss = torch.nn.CosineSimilarity()
        loss_sum = 0
        for encoder_feature, decoder_feature in zip(encoder_features, decoder_features, strict=True):
            loss_sum += torch.mean(
                1
                - cos_loss(
                    encoder_feature.view(encoder_feature.shape[0], -1),
                    decoder_feature.view(decoder_feature.shape[0], -1),
                ),
            )
        return loss_sum
