"""Loss function for the DSR model implementation."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from kornia.losses import FocalLoss
from torch import Tensor, nn


class DsrSecondLoss(nn.Module):
    """Overall loss function of the DSR model.

    The total loss consists of:
        - MSE loss between non-anomalous quantized input image and anomalous subspace-reconstructed non-quantized input (hi and lo)
        - MSE loss between input image and reconstructed image through object-specific decoder,
        - Focal loss between computed segmentation mask and ground truth mask.
    """

    def __init__(self) -> None:
        super().__init__()

        self.l2_loss = nn.modules.loss.MSELoss()
        self.focal_loss = FocalLoss(alpha=1, reduction="mean")

    def forward(self, recon_nq_hi, recon_nq_lo, qu_hi, qu_lo, input_image, gen_img, seg, anomaly_mask) -> Tensor:
        """Compute the loss over a batch for the DSR model."""
        l2_loss_hi_val = self.l2_loss(recon_nq_hi, qu_hi)
        l2_loss_lo_val = self.l2_loss(recon_nq_lo, qu_lo)
        l2_loss_img_val = self.l2_loss(input_image, gen_img)*10
        focal_loss_val = self.focal_loss(seg, anomaly_mask.squeeze(1).long())
        print("hi loss: ", str(l2_loss_hi_val))
        print("lo loss: ", str(l2_loss_lo_val))
        print("img loss: ", str(l2_loss_img_val))
        print("focal loss", str(focal_loss_val))
        return l2_loss_hi_val + l2_loss_lo_val + l2_loss_img_val + focal_loss_val
    
class DsrThirdLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.focal_loss = FocalLoss(alpha=1, reduction="mean")

    def forward(self, truc):
        focal_loss = self.focal_loss()
        print("focal loss", str(focal_loss))
        return focal_loss
