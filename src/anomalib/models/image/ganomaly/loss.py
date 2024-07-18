"""Loss function for the GANomaly Model Implementation."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn


class GeneratorLoss(nn.Module):
    """Generator loss for the GANomaly model.

    Args:
        wadv (int, optional): Weight for adversarial loss.
            Defaults to ``1``.
        wcon (int, optional): Image regeneration weight.
            Defaults to ``50``.
        wenc (int, optional): Latent vector encoder weight.
            Defaults to ``1``.
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
        """Compute the loss for a batch.

        Args:
            latent_i (torch.Tensor): Latent features of the first encoder.
            latent_o (torch.Tensor): Latent features of the second encoder.
            images (torch.Tensor): Real image that served as input of the generator.
            fake (torch.Tensor): Generated image.
            pred_real (torch.Tensor): Discriminator predictions for the real image.
            pred_fake (torch.Tensor): Discriminator predictions for the fake image.

        Returns:
            Tensor: The computed generator loss.
        """
        error_enc = self.loss_enc(latent_i, latent_o)
        error_con = self.loss_con(images, fake)
        error_adv = self.loss_adv(pred_real, pred_fake)

        return error_adv * self.wadv + error_con * self.wcon + error_enc * self.wenc


class DiscriminatorLoss(nn.Module):
    """Discriminator loss for the GANomaly model."""

    def __init__(self) -> None:
        super().__init__()

        self.loss_bce = nn.BCELoss()

    def forward(self, pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
        """Compute the loss for a predicted batch.

        Args:
            pred_real (torch.Tensor): Discriminator predictions for the real image.
            pred_fake (torch.Tensor): Discriminator predictions for the fake image.

        Returns:
            Tensor: The computed discriminator loss.
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
