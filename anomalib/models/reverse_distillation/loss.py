"""Loss function for Reverse Distillation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch
from torch import Tensor


class ReverseDistillationLoss:
    """Loss function for Reverse Distillation."""

    def __call__(self, encoder_features: List[Tensor], decoder_features: List[Tensor]) -> Tensor:
        """Computes cosine similarity loss based on features from encoder and decoder.

        Args:
            encoder_features (List[Tensor]): List of features extracted from encoder
            decoder_features (List[Tensor]): List of features extracted from decoder

        Returns:
            Tensor: Cosine similarity loss
        """
        cos_loss = torch.nn.CosineSimilarity()
        losses = list(map(cos_loss, encoder_features, decoder_features))
        loss_sum = 0
        for loss in losses:
            loss_sum += torch.mean(1 - loss)  # mean of cosine distance
        return loss_sum
