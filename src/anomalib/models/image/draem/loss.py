"""Loss function for the DRAEM model implementation.

This module implements the loss function used to train the DRAEM model for anomaly
detection. The loss combines L2 reconstruction loss, focal loss for anomaly
segmentation, and structural similarity (SSIM) loss.

Example:
    >>> import torch
    >>> from anomalib.models.image.draem.loss import DraemLoss
    >>> criterion = DraemLoss()
    >>> input_image = torch.randn(8, 3, 256, 256)
    >>> reconstruction = torch.randn(8, 3, 256, 256)
    >>> anomaly_mask = torch.randint(0, 2, (8, 1, 256, 256))
    >>> prediction = torch.randn(8, 2, 256, 256)
    >>> loss = criterion(input_image, reconstruction, anomaly_mask, prediction)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from kornia.losses import FocalLoss, SSIMLoss
from torch import nn


class DraemLoss(nn.Module):
    """Overall loss function of the DRAEM model.

    The total loss consists of three components:
    1. L2 loss between the reconstructed and input images
    2. Focal loss between predicted and ground truth anomaly masks
    3. Structural Similarity (SSIM) loss between reconstructed and input images

    The final loss is computed as: ``loss = l2_loss + ssim_loss + focal_loss``

    Example:
        >>> criterion = DraemLoss()
        >>> loss = criterion(input_image, reconstruction, anomaly_mask, prediction)
    """

    def __init__(self) -> None:
        """Initialize loss components with default parameters."""
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
        """Compute the combined loss over a batch for the DRAEM model.

        Args:
            input_image: Original input images of shape
                ``(batch_size, num_channels, height, width)``
            reconstruction: Reconstructed images from the model of shape
                ``(batch_size, num_channels, height, width)``
            anomaly_mask: Ground truth anomaly masks of shape
                ``(batch_size, 1, height, width)``
            prediction: Model predictions of shape
                ``(batch_size, num_classes, height, width)``

        Returns:
            torch.Tensor: Combined loss value
        """
        l2_loss_val = self.l2_loss(reconstruction, input_image)
        focal_loss_val = self.focal_loss(prediction, anomaly_mask.squeeze(1).long())
        ssim_loss_val = self.ssim_loss(reconstruction, input_image) * 2
        return l2_loss_val + ssim_loss_val + focal_loss_val
