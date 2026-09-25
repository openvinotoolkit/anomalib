# Original Code
# Copyright (c) 2024-2026 Niccolò Ferrari
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Residual networks used by GRD-Net."""

from collections.abc import Sequence

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Residual block with an optional projection shortcut.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride of the first convolution and projection shortcut.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.projection = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the residual block.

        Args:
            inputs: Input feature tensor.

        Returns:
            Output feature tensor.
        """
        residual = self.projection(inputs)
        features = self.activation(self.bn1(self.conv1(inputs)))
        features = self.bn2(self.conv2(features))
        return self.activation(features + residual)


class ResidualEncoder(nn.Module):
    """Residual encoder shared by the generator and discriminator designs.

    Args:
        in_channels: Number of input channels.
        base_features: Channel width of the first stage.
        stage_blocks: Number of residual blocks in each stage.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_features: int = 64,
        stage_blocks: Sequence[int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_features),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        blocks: list[nn.Module] = []
        channels = base_features
        for stage_index, block_count in enumerate(stage_blocks):
            stage_channels = base_features * (2**stage_index)
            for block_index in range(block_count):
                stride = 2 if block_index == block_count - 1 else 1
                blocks.append(ResidualBlock(channels, stage_channels, stride=stride))
                channels = stage_channels
        self.blocks = nn.Sequential(*blocks)
        self.out_channels = channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode images into residual features.

        Args:
            inputs: Image tensor.

        Returns:
            Encoded feature tensor.
        """
        return self.blocks(self.stem(inputs))


class ResidualDecoder(nn.Module):
    """Residual decoder that mirrors :class:`ResidualEncoder`.

    Args:
        out_channels: Number of reconstructed image channels.
        base_features: Channel width of the final decoder stage.
        stage_blocks: Number of blocks in each encoder stage.
        bottleneck_channels: Number of input bottleneck channels.
    """

    def __init__(
        self,
        out_channels: int = 3,
        base_features: int = 64,
        stage_blocks: Sequence[int] = (2, 2, 2, 2),
        bottleneck_channels: int = 512,
    ) -> None:
        super().__init__()
        reversed_stages = tuple(reversed(stage_blocks))
        channels = bottleneck_channels
        blocks: list[nn.Module] = []
        for stage_index, block_count in enumerate(reversed_stages):
            target_channels = base_features * (2 ** (len(reversed_stages) - stage_index - 1))
            for block_index in range(block_count):
                if block_index == block_count - 1:
                    blocks.append(
                        nn.Sequential(
                            nn.ConvTranspose2d(
                                channels,
                                target_channels,
                                kernel_size=4,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(target_channels),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        ),
                    )
                else:
                    blocks.append(ResidualBlock(channels, target_channels))
                channels = target_channels

        self.blocks = nn.Sequential(*blocks)
        self.output = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Decode bottleneck features into an image reconstruction.

        Args:
            inputs: Bottleneck feature tensor.

        Returns:
            Reconstructed image tensor.
        """
        return self.output(self.blocks(inputs))


class GRDNetGenerator(nn.Module):
    """Encoder-decoder-encoder generator used by GRD-Net.

    The two encoders and their latent projections are independently parameterized.

    Args:
        in_channels: Number of image channels.
        base_features: Channel width of the first residual stage.
        stage_blocks: Number of residual blocks in each stage.
        latent_channels: Number of channels in each latent feature map.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_features: int = 64,
        stage_blocks: Sequence[int] = (2, 2, 2, 2),
        latent_channels: int = 32,
    ) -> None:
        super().__init__()
        self.encoder = ResidualEncoder(in_channels, base_features, stage_blocks)
        self.latent_projection = nn.Sequential(
            nn.Conv2d(self.encoder.out_channels, latent_channels, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.inverse_projection = nn.Sequential(
            nn.Conv2d(latent_channels, self.encoder.out_channels, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.decoder = ResidualDecoder(
            out_channels=in_channels,
            base_features=base_features,
            stage_blocks=stage_blocks,
            bottleneck_channels=self.encoder.out_channels,
        )
        self.reconstruction_encoder = ResidualEncoder(in_channels, base_features, stage_blocks)
        self.reconstruction_projection = nn.Sequential(
            nn.Conv2d(self.reconstruction_encoder.out_channels, latent_channels, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def reconstruct(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode inputs and reconstruct them through the decoder.

        Args:
            inputs: Input image tensor.

        Returns:
            Tuple containing the latent feature map and image reconstruction.
        """
        latent = self.latent_projection(self.encoder(inputs))
        reconstruction = self.decoder(self.inverse_projection(latent))
        return latent, reconstruction

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct inputs and encode the reconstruction.

        Args:
            inputs: Input image tensor.

        Returns:
            Tuple containing the input latent map, reconstruction, and reconstruction latent map.
        """
        latent, reconstruction = self.reconstruct(inputs)
        reconstruction_latent = self.reconstruction_projection(self.reconstruction_encoder(reconstruction))
        return latent, reconstruction, reconstruction_latent


class GRDNetDiscriminator(nn.Module):
    """Residual adversarial discriminator used by GRD-Net.

    Args:
        input_size: Spatial size of discriminator inputs.
        in_channels: Number of image channels.
        base_features: Channel width of the first residual stage.
        stage_blocks: Number of residual blocks in each stage.

    Raises:
        ValueError: If an input dimension is not divisible by the encoder reduction factor.
    """

    def __init__(
        self,
        input_size: tuple[int, int] = (128, 128),
        in_channels: int = 3,
        base_features: int = 64,
        stage_blocks: Sequence[int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__()
        reduction = 2 ** len(stage_blocks)
        if any(dimension % reduction for dimension in input_size):
            msg = f"Input size {input_size} must be divisible by the encoder reduction factor {reduction}."
            raise ValueError(msg)

        self.encoder = ResidualEncoder(in_channels, base_features, stage_blocks)
        feature_height, feature_width = (dimension // reduction for dimension in input_size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.encoder.out_channels * feature_height * feature_width, 1),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract residual features and predict real/fake logits.

        Args:
            inputs: Input image tensor.

        Returns:
            Tuple containing the final residual feature tensor and real/fake logits.
        """
        features = self.encoder(inputs)
        return features, self.classifier(features)
