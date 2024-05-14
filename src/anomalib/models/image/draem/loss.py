"""Loss function for the DRAEM model implementation."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from kornia.losses import FocalLoss, SSIMLoss
from torch import nn


class DraemLoss(nn.Module):
    """Overall loss function of the DRAEM model.

    The total loss consists of the sum of the L2 loss and Focal loss between the reconstructed image and the input
    image, and the Structural Similarity loss between the predicted and GT anomaly masks.
    """

    def __init__(self) -> None:
        super().__init__()

        self.l2_loss = nn.modules.loss.MSELoss()
        self.focal_loss = FocalLoss(alpha=1, reduction="mean")
        self.ssim_loss = SSIMLoss(window_size=11)

    def forward(
        self,
        input_image: torch.Tensor,
        reconstruction: torch.Tensor,
        anomaly_mask: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the loss over a batch for the DRAEM model."""
        l2_loss_val = self.l2_loss(reconstruction, input_image)
        focal_loss_val = self.focal_loss(prediction, anomaly_mask.squeeze(1).long())
        ssim_loss_val = self.ssim_loss(reconstruction, input_image) * 2
        return l2_loss_val + ssim_loss_val + focal_loss_val
