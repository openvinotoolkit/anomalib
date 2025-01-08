"""Loss functions for the DSR model implementation.

This module contains the loss functions used in the second and third training
phases of the DSR model.

Example:
    >>> from anomalib.models.image.dsr.loss import DsrSecondStageLoss
    >>> loss_fn = DsrSecondStageLoss()
    >>> loss = loss_fn(
    ...     recon_nq_hi=recon_nq_hi,
    ...     recon_nq_lo=recon_nq_lo,
    ...     qu_hi=qu_hi,
    ...     qu_lo=qu_lo,
    ...     input_image=input_image,
    ...     gen_img=gen_img,
    ...     seg=seg,
    ...     anomaly_mask=anomaly_mask
    ... )
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from kornia.losses import FocalLoss
from torch import Tensor, nn


class DsrSecondStageLoss(nn.Module):
    """Loss function for the second training phase of the DSR model.

    The total loss is a combination of:
        - MSE loss between non-anomalous quantized input image and anomalous
          subspace-reconstructed non-quantized input (hi and lo features)
        - MSE loss between input image and reconstructed image through
          object-specific decoder
        - Focal loss between computed segmentation mask and ground truth mask

    Example:
        >>> loss_fn = DsrSecondStageLoss()
        >>> loss = loss_fn(
        ...     recon_nq_hi=recon_nq_hi,
        ...     recon_nq_lo=recon_nq_lo,
        ...     qu_hi=qu_hi,
        ...     qu_lo=qu_lo,
        ...     input_image=input_image,
        ...     gen_img=gen_img,
        ...     seg=seg,
        ...     anomaly_mask=anomaly_mask
        ... )
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
        """Compute the combined loss over a batch.

        Args:
            recon_nq_hi (Tensor): Reconstructed non-quantized hi feature
            recon_nq_lo (Tensor): Reconstructed non-quantized lo feature
            qu_hi (Tensor): Non-defective quantized hi feature
            qu_lo (Tensor): Non-defective quantized lo feature
            input_image (Tensor): Original input image
            gen_img (Tensor): Object-specific decoded image
            seg (Tensor): Computed anomaly segmentation map
            anomaly_mask (Tensor): Ground truth anomaly mask

        Returns:
            Tensor: Total combined loss value

        Example:
            >>> loss_fn = DsrSecondStageLoss()
            >>> loss = loss_fn(
            ...     recon_nq_hi=torch.randn(32, 64, 32, 32),
            ...     recon_nq_lo=torch.randn(32, 64, 32, 32),
            ...     qu_hi=torch.randn(32, 64, 32, 32),
            ...     qu_lo=torch.randn(32, 64, 32, 32),
            ...     input_image=torch.randn(32, 3, 256, 256),
            ...     gen_img=torch.randn(32, 3, 256, 256),
            ...     seg=torch.randn(32, 2, 256, 256),
            ...     anomaly_mask=torch.randint(0, 2, (32, 1, 256, 256))
            ... )
        """
        l2_loss_hi_val = self.l2_loss(recon_nq_hi, qu_hi)
        l2_loss_lo_val = self.l2_loss(recon_nq_lo, qu_lo)
        l2_loss_img_val = self.l2_loss(input_image, gen_img) * 10
        focal_loss_val = self.focal_loss(seg, anomaly_mask.squeeze(1).long())
        return l2_loss_hi_val + l2_loss_lo_val + l2_loss_img_val + focal_loss_val


class DsrThirdStageLoss(nn.Module):
    """Loss function for the third training phase of the DSR model.

    The loss consists of a focal loss between the computed segmentation mask
    and the ground truth mask.

    Example:
        >>> loss_fn = DsrThirdStageLoss()
        >>> loss = loss_fn(
        ...     pred_mask=pred_mask,
        ...     true_mask=true_mask
        ... )
    """

    def __init__(self) -> None:
        super().__init__()

        self.focal_loss = FocalLoss(alpha=1, reduction="mean")

    def forward(self, pred_mask: Tensor, true_mask: Tensor) -> Tensor:
        """Compute the focal loss between predicted and true masks.

        Args:
            pred_mask (Tensor): Computed anomaly segmentation map
            true_mask (Tensor): Ground truth anomaly mask

        Returns:
            Tensor: Focal loss value

        Example:
            >>> loss_fn = DsrThirdStageLoss()
            >>> loss = loss_fn(
            ...     pred_mask=torch.randn(32, 2, 256, 256),
            ...     true_mask=torch.randint(0, 2, (32, 1, 256, 256))
            ... )
        """
        return self.focal_loss(pred_mask, true_mask.squeeze(1).long())
