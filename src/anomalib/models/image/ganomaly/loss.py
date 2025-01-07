"""Loss functions for the GANomaly model implementation.

The GANomaly model uses two loss functions:

1. Generator Loss: Combines adversarial loss, reconstruction loss and encoding loss
2. Discriminator Loss: Binary cross entropy loss for real/fake image discrimination

Example:
    >>> from anomalib.models.image.ganomaly.loss import GeneratorLoss
    >>> generator_loss = GeneratorLoss(wadv=1, wcon=50, wenc=1)
    >>> loss = generator_loss(latent_i, latent_o, images, fake, pred_real, pred_fake)

    >>> from anomalib.models.image.ganomaly.loss import DiscriminatorLoss
    >>> discriminator_loss = DiscriminatorLoss()
    >>> loss = discriminator_loss(pred_real, pred_fake)

See Also:
    :class:`anomalib.models.image.ganomaly.torch_model.GanomalyModel`:
        PyTorch implementation of the GANomaly model architecture.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn


class GeneratorLoss(nn.Module):
    """Generator loss for the GANomaly model.

    Combines three components:
    1. Adversarial loss: Helps generate realistic images
    2. Contextual loss: Ensures generated images match input
    3. Encoding loss: Enforces consistency in latent space

    Args:
        wadv (int, optional): Weight for adversarial loss. Defaults to ``1``.
        wcon (int, optional): Weight for contextual/reconstruction loss.
            Defaults to ``50``.
        wenc (int, optional): Weight for encoding/latent loss. Defaults to ``1``.

    Example:
        >>> generator_loss = GeneratorLoss(wadv=1, wcon=50, wenc=1)
        >>> loss = generator_loss(
        ...     latent_i=torch.randn(32, 100),
        ...     latent_o=torch.randn(32, 100),
        ...     images=torch.randn(32, 3, 256, 256),
        ...     fake=torch.randn(32, 3, 256, 256),
        ...     pred_real=torch.randn(32, 1),
        ...     pred_fake=torch.randn(32, 1)
        ... )
    """

    def __init__(self, wadv: int = 1, wcon: int = 50, wenc: int = 1) -> None:
        super().__init__()

        self.loss_enc = nn.SmoothL1Loss()
        self.loss_adv = nn.MSELoss()
        self.loss_con = nn.L1Loss()

        self.wadv = wadv
        self.wcon = wcon
        self.wenc = wenc

    def forward(
        self,
        latent_i: torch.Tensor,
        latent_o: torch.Tensor,
        images: torch.Tensor,
        fake: torch.Tensor,
        pred_real: torch.Tensor,
        pred_fake: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the generator loss for a batch.

        Args:
            latent_i (torch.Tensor): Latent features from the first encoder.
            latent_o (torch.Tensor): Latent features from the second encoder.
            images (torch.Tensor): Real images that served as generator input.
            fake (torch.Tensor): Generated/fake images.
            pred_real (torch.Tensor): Discriminator predictions for real images.
            pred_fake (torch.Tensor): Discriminator predictions for fake images.

        Returns:
            torch.Tensor: Combined weighted generator loss.

        Example:
            >>> loss = generator_loss(latent_i, latent_o, images, fake,
            ...                      pred_real, pred_fake)
        """
        error_enc = self.loss_enc(latent_i, latent_o)
        error_con = self.loss_con(images, fake)
        error_adv = self.loss_adv(pred_real, pred_fake)

        return error_adv * self.wadv + error_con * self.wcon + error_enc * self.wenc


class DiscriminatorLoss(nn.Module):
    """Discriminator loss for the GANomaly model.

    Uses binary cross entropy to train the discriminator to distinguish between
    real and generated images.

    Example:
        >>> discriminator_loss = DiscriminatorLoss()
        >>> loss = discriminator_loss(
        ...     pred_real=torch.randn(32, 1),
        ...     pred_fake=torch.randn(32, 1)
        ... )
    """

    def __init__(self) -> None:
        super().__init__()

        self.loss_bce = nn.BCELoss()

    def forward(self, pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
        """Compute the discriminator loss for predicted batch.

        Args:
            pred_real (torch.Tensor): Discriminator predictions for real images.
            pred_fake (torch.Tensor): Discriminator predictions for fake images.

        Returns:
            torch.Tensor: Average discriminator loss.

        Example:
            >>> loss = discriminator_loss(pred_real, pred_fake)
        """
        error_discriminator_real = self.loss_bce(
            pred_real,
            torch.ones(size=pred_real.shape, dtype=torch.float32, device=pred_real.device),
        )
        error_discriminator_fake = self.loss_bce(
            pred_fake,
            torch.zeros(size=pred_fake.shape, dtype=torch.float32, device=pred_fake.device),
        )
        return (error_discriminator_fake + error_discriminator_real) * 0.5
