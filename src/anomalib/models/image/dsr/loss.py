"""Loss function for the DSR model implementation."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from kornia.losses import FocalLoss
from torch import Tensor, nn


class DsrSecondStageLoss(nn.Module):
    """Overall loss function of the second training phase of the DSR model.

    The total loss consists of:
        - MSE loss between non-anomalous quantized input image and anomalous subspace-reconstructed
          non-quantized input (hi and lo)
        - MSE loss between input image and reconstructed image through object-specific decoder,
        - Focal loss between computed segmentation mask and ground truth mask.
    """

    def __init__(self) -> None:
        super().__init__()

        self.l2_loss = nn.modules.loss.MSELoss()
        self.focal_loss = FocalLoss(alpha=1, reduction="mean")

    def forward(
        self,
        recon_nq_hi: Tensor,
        recon_nq_lo: Tensor,
        qu_hi: Tensor,
        qu_lo: Tensor,
        input_image: Tensor,
        gen_img: Tensor,
        seg: Tensor,
        anomaly_mask: Tensor,
    ) -> Tensor:
        """Compute the loss over a batch for the DSR model.

        Args:
            recon_nq_hi (Tensor): Reconstructed non-quantized hi feature
            recon_nq_lo (Tensor): Reconstructed non-quantized lo feature
            qu_hi (Tensor): Non-defective quantized hi feature
            qu_lo (Tensor): Non-defective quantized lo feature
            input_image (Tensor): Original image
            gen_img (Tensor): Object-specific decoded image
            seg (Tensor): Computed anomaly map
            anomaly_mask (Tensor): Ground truth anomaly map

        Returns:
            Tensor: Total loss
        """
        l2_loss_hi_val = self.l2_loss(recon_nq_hi, qu_hi)
        l2_loss_lo_val = self.l2_loss(recon_nq_lo, qu_lo)
        l2_loss_img_val = self.l2_loss(input_image, gen_img) * 10
        focal_loss_val = self.focal_loss(seg, anomaly_mask.squeeze(1).long())
        return l2_loss_hi_val + l2_loss_lo_val + l2_loss_img_val + focal_loss_val


class DsrThirdStageLoss(nn.Module):
    """Overall loss function of the third training phase of the DSR model.

    The loss consists of a focal loss between the computed segmentation mask and the ground truth mask.
    """

    def __init__(self) -> None:
        super().__init__()

        self.focal_loss = FocalLoss(alpha=1, reduction="mean")

    def forward(self, pred_mask: Tensor, true_mask: Tensor) -> Tensor:
        """Compute the loss over a batch for the DSR model.

        Args:
            pred_mask (Tensor): Computed anomaly map
            true_mask (Tensor): Ground truth anomaly map

        Returns:
            Tensor: Total loss
        """
        return self.focal_loss(pred_mask, true_mask.squeeze(1).long())
