"""Loss function for Reverse Distillation."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import geomloss
import torch
from torch import nn


class ProjLayer(nn.Module):
    """Projection Layer for feature transformation.

    Args:
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.
    """

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_c, in_c // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_c // 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_c // 2, in_c // 4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_c // 4),
            nn.LeakyReLU(),
            nn.Conv2d(in_c // 4, in_c // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_c // 2),
            nn.LeakyReLU(),
            nn.Conv2d(in_c // 2, out_c, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the projection layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        return self.proj(x)


class MultiProjectionLayer(nn.Module):
    """Multi-Projection Layer for handling multiple feature maps."""

    def __init__(self, base: int = 64) -> None:
        super().__init__()
        self.proj_a = ProjLayer(base * 4, base * 4)
        self.proj_b = ProjLayer(base * 8, base * 8)
        self.proj_c = ProjLayer(base * 16, base * 16)

    def forward(
        self,
        features: list[torch.Tensor],
        features_noise: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass for the multi-projection layer.

        Args:
            features (list[torch.Tensor]): List of feature maps.
            features_noise (list[torch.Tensor], optional): List of noisy feature maps. Defaults to None.

        Returns:
            list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]: Transformed feature maps.
        """
        if features_noise is not None:
            return (
                [self.proj_a(features_noise[0]), self.proj_b(features_noise[1]), self.proj_c(features_noise[2])],
                [self.proj_a(features[0]), self.proj_b(features[1]), self.proj_c(features[2])],
            )
        return [self.proj_a(features[0]), self.proj_b(features[1]), self.proj_c(features[2])]


# src/anomalib/models/image/revisiting_rd/loss.py


class CosineReconstruct(nn.Module):
    """Cosine Reconstruction Loss."""

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the cosine reconstruction loss.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: Cosine reconstruction loss.
        """
        return torch.mean(1 - torch.nn.CosineSimilarity()(x, y))


class RevisitingReverseDistillationLoss(nn.Module):
    """Loss function for Reverse Distillation."""

    def __init__(self) -> None:
        super().__init__()
        self.sinkhorn = geomloss.SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=0.05,
            reach=None,
            diameter=10000000,
            scaling=0.95,
            truncate=10,
            cost=None,
            kernel=None,
            cluster_scale=None,
            debias=True,
            potentials=False,
            verbose=False,
            backend="auto",
        )
        self.reconstruct = CosineReconstruct()
        self.contrast = torch.nn.CosineEmbeddingLoss(margin=0.5)

    def forward(
        self,
        encoder_features: list[torch.Tensor],
        decoder_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the combined loss based on features from encoder and decoder.

        Args:
            encoder_features (list[torch.Tensor]): List of features extracted from encoder.
            decoder_features (list[torch.Tensor]): List of features extracted from decoder.

        Returns:
            torch.Tensor: Combined loss.
        """
        current_batchsize = encoder_features[0].shape[0]
        target = -torch.ones(current_batchsize).to("cuda")

        normal_proj1, normal_proj2, normal_proj3 = decoder_features
        shuffle_index = torch.randperm(current_batchsize)
        shuffle_1 = normal_proj1[shuffle_index]
        shuffle_2 = normal_proj2[shuffle_index]
        shuffle_3 = normal_proj3[shuffle_index]

        abnormal_proj1, abnormal_proj2, abnormal_proj3 = encoder_features

        # Compute SSOT loss
        loss_ssot = (
            self.sinkhorn(
                torch.softmax(normal_proj1.view(normal_proj1.shape[0], -1), -1),
                torch.softmax(shuffle_1.view(shuffle_1.shape[0], -1), -1),
            )
            + self.sinkhorn(
                torch.softmax(normal_proj2.view(normal_proj2.shape[0], -1), -1),
                torch.softmax(shuffle_2.view(shuffle_2.shape[0], -1), -1),
            )
            + self.sinkhorn(
                torch.softmax(normal_proj3.view(normal_proj3.shape[0], -1), -1),
                torch.softmax(shuffle_3.view(shuffle_3.shape[0], -1), -1),
            )
        )

        # Compute reconstruction loss
        loss_reconstruct = (
            self.reconstruct(abnormal_proj1, normal_proj1)
            + self.reconstruct(abnormal_proj2, normal_proj2)
            + self.reconstruct(abnormal_proj3, normal_proj3)
        )

        # Compute contrastive loss
        loss_contrast = (
            self.contrast(
                abnormal_proj1.view(abnormal_proj1.shape[0], -1),
                normal_proj1.view(normal_proj1.shape[0], -1),
                target=target,
            )
            + self.contrast(
                abnormal_proj2.view(abnormal_proj2.shape[0], -1),
                normal_proj2.view(normal_proj2.shape[0], -1),
                target=target,
            )
            + self.contrast(
                abnormal_proj3.view(abnormal_proj3.shape[0], -1),
                normal_proj3.view(normal_proj3.shape[0], -1),
                target=target,
            )
        )

        # Compute additional loss using loss_function
        additional_loss = loss_function(encoder_features, decoder_features)

        # Combine losses with specific weights
        return (loss_ssot + 0.01 * loss_reconstruct + 0.1 * loss_contrast + additional_loss) / 1.11


def loss_function(a: list[torch.Tensor], b: list[torch.Tensor]) -> torch.Tensor:
    """Cosine similarity loss function.

    Args:
        a (list[torch.Tensor]): List of tensors from the encoder.
        b (list[torch.Tensor]): List of tensors from the decoder.

    Returns:
        torch.Tensor: Cosine similarity loss.
    """
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1), b[item].view(b[item].shape[0], -1)))
    return loss
