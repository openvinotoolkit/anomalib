# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Loss functions for GRD-Net training."""

from typing import NamedTuple

import torch
from kornia.losses import FocalLoss, SSIMLoss
from torch import nn


class GeneratorLosses(NamedTuple):
    """GRD-Net generator loss terms.

    Attributes:
        total: Weighted sum of the generator objectives.
        adversarial: Discriminator feature-matching loss.
        contextual: Image reconstruction and structural-similarity loss.
        encoder: Latent-consistency loss.
    """

    total: torch.Tensor
    adversarial: torch.Tensor
    contextual: torch.Tensor
    encoder: torch.Tensor


class GRDNetGeneratorLoss(nn.Module):
    """Compute the weighted GRD-Net generator objective.

    Args:
        adversarial_weight: Weight applied to discriminator feature matching.
        contextual_weight: Weight applied to image reconstruction.
        encoder_weight: Weight applied to latent consistency.
    """

    def __init__(
        self,
        adversarial_weight: float = 1.0,
        contextual_weight: float = 50.0,
        encoder_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.adversarial_weight = adversarial_weight
        self.contextual_weight = contextual_weight
        self.encoder_weight = encoder_weight
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss(window_size=11)

    def forward(
        self,
        image: torch.Tensor,
        reconstruction: torch.Tensor,
        latent: torch.Tensor,
        reconstruction_latent: torch.Tensor,
        real_features: torch.Tensor,
        fake_features: torch.Tensor,
    ) -> GeneratorLosses:
        """Compute all generator loss terms.

        Args:
            image: Clean target image.
            reconstruction: Image reconstructed by the generator.
            latent: Latent representation of the generator input.
            reconstruction_latent: Latent representation of the reconstruction.
            real_features: Discriminator features for the clean image.
            fake_features: Discriminator features for the reconstruction.

        Returns:
            GeneratorLosses: Total loss and its three unweighted components.
        """
        contextual = self.l1_loss(reconstruction, image) + (2 * self.ssim_loss(reconstruction, image))
        adversarial = self.mse_loss(fake_features, real_features.detach())
        encoder = self.l1_loss(reconstruction_latent, latent)
        total = (
            self.adversarial_weight * adversarial + self.contextual_weight * contextual + self.encoder_weight * encoder
        )
        return GeneratorLosses(total, adversarial, contextual, encoder)


class GRDNetDiscriminatorLoss(nn.Module):
    """Compute the GRD-Net real/fake discriminator objective."""

    def __init__(self) -> None:
        super().__init__()
        self.binary_cross_entropy = nn.BCEWithLogitsLoss()

    def forward(self, real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
        """Compute the mean real and fake binary cross-entropy.

        Args:
            real_logits: Discriminator logits for real images.
            fake_logits: Discriminator logits for detached reconstructions.

        Returns:
            torch.Tensor: Scalar discriminator loss.
        """
        real_loss = self.binary_cross_entropy(real_logits, torch.ones_like(real_logits))
        fake_loss = self.binary_cross_entropy(fake_logits, torch.zeros_like(fake_logits))
        return (real_loss + fake_loss) * 0.5


class GRDNetSegmentatorLoss(nn.Module):
    """Compute focal loss against the ROI-intersected anomaly target."""

    def __init__(self) -> None:
        super().__init__()
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0, reduction="mean")

    @staticmethod
    def target(anomaly_mask: torch.Tensor, roi_mask: torch.Tensor) -> torch.Tensor:
        """Intersect the synthetic anomaly mask with its training ROI.

        Args:
            anomaly_mask: Binary synthetic anomaly mask with a singleton channel.
            roi_mask: Binary training ROI mask with a singleton channel.

        Returns:
            torch.Tensor: Integer class target without the channel dimension.
        """
        return (anomaly_mask * roi_mask).squeeze(1).long()

    def forward(
        self,
        logits: torch.Tensor,
        anomaly_mask: torch.Tensor,
        roi_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multiclass focal loss for segmentator logits.

        Args:
            logits: Two-class segmentator logits.
            anomaly_mask: Binary synthetic anomaly mask.
            roi_mask: Binary training ROI mask.

        Returns:
            torch.Tensor: Scalar focal loss.
        """
        return self.focal_loss(logits, self.target(anomaly_mask, roi_mask))
