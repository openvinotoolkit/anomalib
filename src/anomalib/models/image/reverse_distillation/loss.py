"""Loss function for Reverse Distillation."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn


class ReverseDistillationLoss(nn.Module):
    """Loss function for Reverse Distillation."""

    @staticmethod
    def forward(encoder_features: list[torch.Tensor], decoder_features: list[torch.Tensor]) -> torch.Tensor:
        """Compute cosine similarity loss based on features from encoder and decoder.

        Based on the official code:
        https://github.com/hq-deng/RD4AD/blob/6554076872c65f8784f6ece8cfb39ce77e1aee12/main.py#L33C25-L33C25
        Calculates loss from flattened arrays of features, see https://github.com/hq-deng/RD4AD/issues/22

        Args:
            encoder_features (list[torch.Tensor]): List of features extracted from encoder
            decoder_features (list[torch.Tensor]): List of features extracted from decoder

        Returns:
            Tensor: Cosine similarity loss
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
