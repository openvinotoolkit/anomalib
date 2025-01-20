"""PyTorch model for the DRAEM model implementation.

The DRAEM model consists of two sub-networks:
1. A reconstructive sub-network that learns to reconstruct input images
2. A discriminative sub-network that detects anomalies by comparing original and
   reconstructed images
"""

# Original Code
# Copyright (c) 2021 VitjanZ
# https://github.com/VitjanZ/DRAEM.
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from anomalib.data import InferenceBatch
from anomalib.models.components.layers import SSPCAB


class DraemModel(nn.Module):
    """DRAEM PyTorch model with reconstructive and discriminative sub-networks.

    Args:
        sspcab (bool, optional): Enable SSPCAB training. Defaults to ``False``.

    Example:
        >>> model = DraemModel(sspcab=True)
        >>> input_tensor = torch.randn(32, 3, 256, 256)
        >>> reconstruction, prediction = model(input_tensor)
    """

    def __init__(self, sspcab: bool = False) -> None:
        super().__init__()
        self.reconstructive_subnetwork = ReconstructiveSubNetwork(sspcab=sspcab)
        self.discriminative_subnetwork = DiscriminativeSubNetwork(in_channels=6, out_channels=2)

    def forward(self, batch: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | InferenceBatch:
        """Forward pass through both sub-networks.

        Args:
            batch (torch.Tensor): Input batch of images of shape
                ``(batch_size, channels, height, width)``

        Returns:
            During training:
                tuple: Tuple containing:
                    - Reconstructed images
                    - Predicted anomaly masks
            During inference:
                InferenceBatch: Contains anomaly map and prediction score

        Example:
            >>> model = DraemModel()
            >>> batch = torch.randn(32, 3, 256, 256)
            >>> reconstruction, prediction = model(batch)  # Training mode
            >>> model.eval()
            >>> output = model(batch)  # Inference mode
            >>> assert isinstance(output, InferenceBatch)
        """
        reconstruction = self.reconstructive_subnetwork(batch)
        concatenated_inputs = torch.cat([batch, reconstruction], axis=1)
        prediction = self.discriminative_subnetwork(concatenated_inputs)
        if self.training:
            return reconstruction, prediction

        anomaly_map = torch.softmax(prediction, dim=1)[:, 1, ...]
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)


class ReconstructiveSubNetwork(nn.Module):
    """Autoencoder model for image reconstruction.

    Args:
        in_channels (int, optional): Number of input channels. Defaults to ``3``.
        out_channels (int, optional): Number of output channels. Defaults to ``3``.
        base_width (int, optional): Base dimensionality of layers. Defaults to
            ``128``.
        sspcab (bool, optional): Enable SSPCAB training. Defaults to ``False``.

    Example:
        >>> subnet = ReconstructiveSubNetwork(in_channels=3, base_width=64)
        >>> input_tensor = torch.randn(32, 3, 256, 256)
        >>> output = subnet(input_tensor)
        >>> output.shape
        torch.Size([32, 3, 256, 256])
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_width: int = 128,
        sspcab: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = EncoderReconstructive(in_channels, base_width, sspcab=sspcab)
        self.decoder = DecoderReconstructive(base_width, out_channels=out_channels)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Encode and reconstruct input images.

        Args:
            batch (torch.Tensor): Batch of input images of shape
                ``(batch_size, channels, height, width)``

        Returns:
            torch.Tensor: Batch of reconstructed images of same shape as input
        """
        encoded = self.encoder(batch)
        return self.decoder(encoded)


class DiscriminativeSubNetwork(nn.Module):
    """Discriminative model for anomaly mask prediction.

    Compares original images with their reconstructions to predict anomaly masks.

    Args:
        in_channels (int, optional): Number of input channels. Defaults to ``3``.
        out_channels (int, optional): Number of output channels. Defaults to ``3``.
        base_width (int, optional): Base dimensionality of layers. Defaults to
            ``64``.

    Example:
        >>> subnet = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
        >>> # Concatenated original and reconstructed images
        >>> input_tensor = torch.randn(32, 6, 256, 256)
        >>> output = subnet(input_tensor)
        >>> output.shape
        torch.Size([32, 2, 256, 256])
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_width: int = 64) -> None:
        super().__init__()
        self.encoder_segment = EncoderDiscriminative(in_channels, base_width)
        self.decoder_segment = DecoderDiscriminative(base_width, out_channels=out_channels)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Generate predicted anomaly masks.

        Args:
            batch (torch.Tensor): Concatenated original and reconstructed images of
                shape ``(batch_size, channels*2, height, width)``

        Returns:
            torch.Tensor: Pixel-level class scores for normal and anomalous regions
        """
        act1, act2, act3, act4, act5, act6 = self.encoder_segment(batch)
        return self.decoder_segment(act1, act2, act3, act4, act5, act6)


class EncoderDiscriminative(nn.Module):
    """Encoder component of the discriminator network.

    Args:
        in_channels (int): Number of input channels
        base_width (int): Base dimensionality of the layers

    Example:
        >>> encoder = EncoderDiscriminative(in_channels=6, base_width=64)
        >>> input_tensor = torch.randn(32, 6, 256, 256)
        >>> outputs = encoder(input_tensor)
        >>> len(outputs)  # Returns 6 activation maps
        6
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

    def forward(
        self,
        batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert inputs to salient space through encoder network.

        Args:
            batch (torch.Tensor): Input batch of concatenated original and
                reconstructed images

        Returns:
            tuple: Contains 6 activation tensors from each encoder block
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
    """Decoder component of the discriminator network.

    Args:
        base_width (int): Base dimensionality of the layers
        out_channels (int, optional): Number of output channels. Defaults to ``1``

    Example:
        >>> decoder = DecoderDiscriminative(base_width=64, out_channels=2)
        >>> # Create 6 mock activation tensors
        >>> acts = [torch.randn(32, 64, 256>>i, 256>>i) for i in range(6)]
        >>> output = decoder(*acts)
        >>> output.shape
        torch.Size([32, 2, 256, 256])
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

    def forward(
        self,
        act1: torch.Tensor,
        act2: torch.Tensor,
        act3: torch.Tensor,
        act4: torch.Tensor,
        act5: torch.Tensor,
        act6: torch.Tensor,
    ) -> torch.Tensor:
        """Compute predicted anomaly scores from encoder activations.

        Args:
            act1 (torch.Tensor): First block encoder activations
            act2 (torch.Tensor): Second block encoder activations
            act3 (torch.Tensor): Third block encoder activations
            act4 (torch.Tensor): Fourth block encoder activations
            act5 (torch.Tensor): Fifth block encoder activations
            act6 (torch.Tensor): Sixth block encoder activations

        Returns:
            torch.Tensor: Predicted anomaly scores per pixel
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

        return self.fin_out(db4)


class EncoderReconstructive(nn.Module):
    """Encoder component of the reconstructive network.

    Args:
        in_channels (int): Number of input channels
        base_width (int): Base dimensionality of the layers
        sspcab (bool, optional): Enable SSPCAB training. Defaults to ``False``

    Example:
        >>> encoder = EncoderReconstructive(in_channels=3, base_width=64)
        >>> input_tensor = torch.randn(32, 3, 256, 256)
        >>> output = encoder(input_tensor)
        >>> output.shape
        torch.Size([32, 512, 16, 16])
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

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Encode input images to the salient space.

        Args:
            batch (torch.Tensor): Input batch of images of shape
                ``(batch_size, channels, height, width)``

        Returns:
            torch.Tensor: Feature maps from the bottleneck layer
        """
        act1 = self.block1(batch)
        mp1 = self.mp1(act1)
        act2 = self.block2(mp1)
        mp2 = self.mp3(act2)
        act3 = self.block3(mp2)
        mp3 = self.mp3(act3)
        act4 = self.block4(mp3)
        mp4 = self.mp4(act4)
        return self.block5(mp4)


class DecoderReconstructive(nn.Module):
    """Decoder component of the reconstructive network.

    Args:
        base_width (int): Base dimensionality of the layers
        out_channels (int, optional): Number of output channels. Defaults to ``1``

    Example:
        >>> decoder = DecoderReconstructive(base_width=64, out_channels=3)
        >>> input_tensor = torch.randn(32, 512, 16, 16)
        >>> output = decoder(input_tensor)
        >>> output.shape
        torch.Size([32, 3, 256, 256])
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

    def forward(self, act5: torch.Tensor) -> torch.Tensor:
        """Reconstruct image from bottleneck features.

        Args:
            act5 (torch.Tensor): Activations from the bottleneck layer of shape
                ``(batch_size, channels, height, width)``

        Returns:
            torch.Tensor: Reconstructed images of same size as original input
        """
        up1 = self.up1(act5)
        db1 = self.db1(up1)

        up2 = self.up2(db1)
        db2 = self.db2(up2)

        up3 = self.up3(db2)
        db3 = self.db3(up3)

        up4 = self.up4(db3)
        db4 = self.db4(up4)

        return self.fin_out(db4)
