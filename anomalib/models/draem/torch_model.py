"""PyTorch model for the DRAEM model implementation."""

# Original Code
# Copyright (c) 2021 VitjanZ
# https://github.com/VitjanZ/DRAEM.
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import Tensor, nn

from anomalib.models.components.layers import SSPCAB


class DraemModel(nn.Module):
    """DRAEM PyTorch model consisting of the reconstructive and discriminative sub networks."""

    def __init__(self, sspcab: bool = False) -> None:
        super().__init__()
        self.reconstructive_subnetwork = ReconstructiveSubNetwork(sspcab=sspcab)
        self.discriminative_subnetwork = DiscriminativeSubNetwork(in_channels=6, out_channels=2)

    def forward(self, batch: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Compute the reconstruction and anomaly mask from an input image.

        Args:
            x (Tensor): batch of input images

        Returns:
            Predicted confidence values of the anomaly mask. During training the reconstructed input images are
            returned as well.
        """
        reconstruction = self.reconstructive_subnetwork(batch)
        concatenated_inputs = torch.cat([batch, reconstruction], axis=1)
        prediction = self.discriminative_subnetwork(concatenated_inputs)
        if self.training:
            return reconstruction, prediction
        return torch.softmax(prediction, dim=1)[:, 1, ...]


class ReconstructiveSubNetwork(nn.Module):
    """Autoencoder model that encodes and reconstructs the input image.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        base_width (int): Base dimensionality of the layers of the autoencoder.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_width=128, sspcab: bool = False) -> None:
        super().__init__()
        self.encoder = EncoderReconstructive(in_channels, base_width, sspcab=sspcab)
        self.decoder = DecoderReconstructive(base_width, out_channels=out_channels)

    def forward(self, batch: Tensor) -> Tensor:
        """Encode and reconstruct the input images.

        Args:
            batch (Tensor): Batch of input images

        Returns:
            Batch of reconstructed images.
        """
        encoded = self.encoder(batch)
        decoded = self.decoder(encoded)
        return decoded


class DiscriminativeSubNetwork(nn.Module):
    """Discriminative model that predicts the anomaly mask from the original image and its reconstruction.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        base_width (int): Base dimensionality of the layers of the autoencoder.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_width: int = 64) -> None:
        super().__init__()
        self.encoder_segment = EncoderDiscriminative(in_channels, base_width)
        self.decoder_segment = DecoderDiscriminative(base_width, out_channels=out_channels)

    def forward(self, batch: Tensor) -> Tensor:
        """Generate the predicted anomaly masks for a batch of input images.

        Args:
            batch (Tensor): Batch of inputs consisting of the concatenation of the original images
             and their reconstructions.

        Returns:
            Activations of the output layer corresponding to the normal and anomalous class scores on the pixel level.
        """
        act1, act2, act3, act4, act5, act6 = self.encoder_segment(batch)
        segmentation = self.decoder_segment(act1, act2, act3, act4, act5, act6)
        return segmentation


class EncoderDiscriminative(nn.Module):
    """Encoder part of the discriminator network.

    Args:
        in_channels (int): Number of input channels.
        base_width (int): Base dimensionality of the layers of the autoencoder.
    """

    def __init__(self, in_channels: int, base_width: int) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )

        self.mp5 = nn.Sequential(nn.MaxPool2d(2))
        self.block6 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )

    def forward(self, batch: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Convert the inputs to the salient space by running them through the encoder network.

        Args:
            batch (Tensor): Batch of inputs consisting of the concatenation of the original images
             and their reconstructions.

        Returns:
            Computed feature maps for each of the layers in the encoder sub network.
        """
        act1 = self.block1(batch)
        mp1 = self.mp1(act1)
        act2 = self.block2(mp1)
        mp2 = self.mp3(act2)
        act3 = self.block3(mp2)
        mp3 = self.mp3(act3)
        act4 = self.block4(mp3)
        mp4 = self.mp4(act4)
        act5 = self.block5(mp4)
        mp5 = self.mp5(act5)
        act6 = self.block6(mp5)
        return act1, act2, act3, act4, act5, act6


class DecoderDiscriminative(nn.Module):
    """Decoder part of the discriminator network.

    Args:
        base_width (int): Base dimensionality of the layers of the autoencoder.
        out_channels (int): Number of output channels.
    """

    def __init__(self, base_width: int, out_channels: int = 1) -> None:
        super().__init__()

        self.up_b = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )
        self.db_b = nn.Sequential(
            nn.Conv2d(base_width * (8 + 8), base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * (4 + 8), base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * (2 + 4), base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * (2 + 1), base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, act1: Tensor, act2: Tensor, act3: Tensor, act4: Tensor, act5: Tensor, act6: Tensor) -> Tensor:
        """Computes predicted anomaly class scores from the intermediate outputs of the encoder sub network.

        Args:
            act1 (Tensor): Encoder activations of the first block of convolutional layers.
            act2 (Tensor): Encoder activations of the second block of convolutional layers.
            act3 (Tensor): Encoder activations of the third block of convolutional layers.
            act4 (Tensor): Encoder activations of the fourth block of convolutional layers.
            act5 (Tensor): Encoder activations of the fifth block of convolutional layers.
            act6 (Tensor): Encoder activations of the sixth block of convolutional layers.

        Returns:
            Predicted anomaly class scores per pixel.
        """
        up_b = self.up_b(act6)
        cat_b = torch.cat((up_b, act5), dim=1)
        db_b = self.db_b(cat_b)

        up1 = self.up1(db_b)
        cat1 = torch.cat((up1, act4), dim=1)
        db1 = self.db1(cat1)

        up2 = self.up2(db1)
        cat2 = torch.cat((up2, act3), dim=1)
        db2 = self.db2(cat2)

        up3 = self.up3(db2)
        cat3 = torch.cat((up3, act2), dim=1)
        db3 = self.db3(cat3)

        up4 = self.up4(db3)
        cat4 = torch.cat((up4, act1), dim=1)
        db4 = self.db4(cat4)

        out = self.fin_out(db4)
        return out


class EncoderReconstructive(nn.Module):
    """Encoder part of the reconstructive network.

    Args:
        in_channels (int): Number of input channels.
        base_width (int): Base dimensionality of the layers of the autoencoder.
    """

    def __init__(self, in_channels: int, base_width: int, sspcab: bool = False) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        if sspcab:
            self.block5 = SSPCAB(base_width * 8)
        else:
            self.block5 = nn.Sequential(
                nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_width * 8),
                nn.ReLU(inplace=True),
                nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(base_width * 8),
                nn.ReLU(inplace=True),
            )

    def forward(self, batch: Tensor) -> Tensor:
        """Encode a batch of input images to the salient space.

        Args:
            batch (Tensor): Batch of input images.

        Returns:
            Feature maps extracted from the bottleneck layer.
        """
        act1 = self.block1(batch)
        mp1 = self.mp1(act1)
        act2 = self.block2(mp1)
        mp2 = self.mp3(act2)
        act3 = self.block3(mp2)
        mp3 = self.mp3(act3)
        act4 = self.block4(mp3)
        mp4 = self.mp4(act4)
        act5 = self.block5(mp4)
        return act5


class DecoderReconstructive(nn.Module):
    """Decoder part of the reconstructive network.

    Args:
        base_width (int): Base dimensionality of the layers of the autoencoder.
        out_channels (int): Number of output channels.
    """

    def __init__(self, base_width: int, out_channels: int = 1) -> None:
        super().__init__()

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
        )
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
        )
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
        )
        # cat with base*1
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 1),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width * 1, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, act5: Tensor) -> Tensor:
        """Reconstruct the image from the activations of the bottleneck layer.

        Args:
            act5 (Tensor): Activations of the bottleneck layer.

        Returns:
            Batch of reconstructed images.
        """
        up1 = self.up1(act5)
        db1 = self.db1(up1)

        up2 = self.up2(db1)
        db2 = self.db2(up2)

        up3 = self.up3(db2)
        db3 = self.db3(up3)

        up4 = self.up4(db3)
        db4 = self.db4(up4)

        out = self.fin_out(db4)
        return out
