"""Loss function for Reverse Distillation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import Tensor, nn


class ReverseDistillationLoss(nn.Module):
    """Loss function for Reverse Distillation."""

    def forward(self, encoder_features: list[Tensor], decoder_features: list[Tensor]) -> Tensor:
        """Computes cosine similarity loss based on features from encoder and decoder.

        Args:
            encoder_features (list[Tensor]): List of features extracted from encoder
            decoder_features (list[Tensor]): List of features extracted from decoder

        Returns:
            Tensor: Cosine similarity loss
        """
        cos_loss = torch.nn.CosineSimilarity()
        losses = list(map(cos_loss, encoder_features, decoder_features))
        loss_sum = 0
        for loss in losses:
            loss_sum += torch.mean(1 - loss)  # mean of cosine distance
        return loss_sum
