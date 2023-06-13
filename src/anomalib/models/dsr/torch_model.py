"""PyTorch model for the DSR model implementation."""

# Original Code
# Copyright (c) 2022 VitjanZ
# https://github.com/VitjanZ/DSR_anomaly_detection.
# SPDX-License-Identifier: Apache-2.0
#
# Modified
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from torch import Tensor, nn, F

from anomalib.models.components.layers import SSPCAB


class DsrModel(nn.Module):
    """DSR PyTorch model consisting of the quantized encoder and decoder, anomaly detection
    module, and upsampling module"""

    def __init__(self, sspcab: bool = False) -> None:
        super().__init__()
        self.quantized_encoder = None
        self.specific_decoder = None
        self.general_decoder = None
        self.anomaly_detection_module = None
        self.upsampling_module = None
        self.anomaly_generation_module = None

    def forward(self, batch: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Compute the reconstruction and anomaly mask from an input image.

        Args:
            x (Tensor): batch of input images

        Returns:
            Predicted confidence values of the anomaly mask. During training the reconstructed input images are
            returned as well.
        """

class SubspaceRestrictionModule(nn.Module):
    """Subspace restriction module that restricts the appearance subspace into configurations
    that agree with normal appearances and applies quantization.
    
    Args:
        base_width (int): Base dimensionality of the layers of the autoencoder.
    """

    def __init__(self, base_width: int = 64):
        super(SubspaceRestrictionModule, self).__init__()

        self.unet = SubspaceRestrictionNetwork(in_channels=base_width, out_channels=base_width, base_width=base_width)

    def forward(self, batch : Tensor, quantization : function | object):
        """Generate the quantized anomaly-free representation of an anomalous image.
        
        Args:
            batch (Tensor): Batch of input images.
            quantization (function | object): Quantization function.
        
        Returns:
            Tuple containing reconstructed batch of non-quantized features, quantized features, and quantization loss.
        """
        batch = self.unet(batch)
        loss_b, quantized_b, perplexity_b, encodings_b = quantization(batch)
        return batch, quantized_b, loss_b

class SubspaceRestrictionNetwork(nn.Module):
    """Subspace restriction network that reconstructs the input image into a
    non-quantized configuration that agrees with normal appearances.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        base_width (int): Base dimensionality of the layers of the autoencoder.
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 64, base_width: int = 64):
        super().__init__()
        self.base_width = base_width
        self.encoder = FeatureEncoder(in_channels, self.base_width)
        self.decoder = FeatureDecoder(self.base_width, out_channels=out_channels)

    def forward(self, batch: Tensor):
        """Generate non-quantized feature maps from potentially anomalous images, to
        be quantized into non-anomalous quantized representations.
        
        Args:
            batch (Tensor): Batch of input images.
        
        Returns:
            Reconstructed non-quantized representation.
        """
        b1, b2, b3 = self.encoder(batch)
        output = self.decoder(b1, b2, b3)
        return output

class FeatureEncoder(nn.Module):
    """Feature encoder for the subspace restriction network.
    
    Args:
        in_channels (int): Number of input channels.
        base_width (int): Base dimensionality of the layers of the autoencoder.
    """

    def __init__(self, in_channels: int, base_width: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 4),
            nn.ReLU(inplace=True))

    def forward(self, batch: Tensor):
        """Encode a batch of input features to the latent space.
        
        Args:
            batch (Tensor): Batch of input images.
        
        Returns:
            Encoded feature maps."""
        b1 = self.block1(batch)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        return b1, b2, b3

class FeatureDecoder(nn.Module):
    """Feature decoder for the subspace restriction network.
    
    Args:
        base_width (int): Base dimensionality of the layers of the autoencoder.
        out_channels (int): Number of output channels.
    """

    def __init__(self, base_width: int, out_channels: int = 1):
        super().__init__()

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
                                 nn.InstanceNorm2d(base_width * 2),
                                 nn.ReLU(inplace=True))

        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
                                 nn.InstanceNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, b1: Tensor, b2: Tensor, b3: Tensor):
        """Decode a batch of latent features to a non-quantized representation.
        
        Args:
            b1 (Tensor): Top latent feature layer.
            b2 (Tensor): Middle latent feature layer.
            b3 (Tensor): Bottom latent feature layer.
        
        Returns:
            Decoded non-quantized representation.
        """
        up2 = self.up2(b3)
        db2 = self.db2(up2)

        up3 = self.up3(db2)
        db3 = self.db3(up3)

        out = self.fin_out(db3)
        return out

class Residual(nn.Module):
    """Residual layer.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_residual_hiddens (int): Number of intermediate channels.
    """

    def __init__(self, in_channels: int, out_channels: int, num_residual_hiddens: int):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=out_channels,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, batch: Tensor):
        """Compute residual layer.
        
        Args:
            batch (Tensor): Batch of input images.
        
        Returns:
            Computed feature maps.
        """
        return batch + self._block(batch)


class ResidualStack(nn.Module):
    """Stack of residual layers.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_residual_layers (int): Number of residual layers.
        num_residual_hiddens (int): Number of intermediate channels.
    """

    def __init__(self, in_channels: int, out_channels: int, num_residual_layers: int, num_residual_hiddens: int):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, out_channels, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, batch: Tensor):
        """Compute residual stack.
        
        Args:
            batch (Tensor): Batch of input images.
        
        Returns:
            Computed feature maps.
        """
        for i in range(self._num_residual_layers):
            batch = self._layers[i](batch)
        return F.relu(batch)


class ImageReconstructionNetwork(nn.Module):
    """Image reconstruction network that reconstructs the image from a quantized
    representation.
    
    Args:
        in_channels (int): Number of input channels.
        num_hiddens (int): Number of output channels in residual layers.
        num_residual_layers (int): Number of residual layers.
        num_residual_hiddens (int): Number of intermediate channels.
    """

    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int):
        super(ImageReconstructionNetwork, self).__init__()
        norm_layer = nn.InstanceNorm2d
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=1),
            norm_layer(in_channels*2),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels * 2, kernel_size=3, padding=1),
            norm_layer(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=3, padding=1),
            norm_layer(in_channels * 4),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))

        self.pre_vq_conv = nn.Conv2d(in_channels=in_channels*4,
                                 out_channels=64,
                                 kernel_size=1,
                                 stride=1)

        self.upblock1 = nn.ConvTranspose2d(in_channels=64,
                                                out_channels=64,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self.upblock2 = nn.ConvTranspose2d(in_channels=64,
                                                out_channels=64,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_1 = nn.Conv2d(in_channels=64,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=3,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs: Tensor):
        """Reconstructs an image from a quantized representation.
        
        Args:
            inputs (Tensor): Quantized features.
        
        Returns:
            Reconstructed image.
        """
        batch = self.block1(inputs)
        batch = self.mp1(batch)
        batch = self.block2(batch)
        batch = self.mp2(batch)
        batch = self.pre_vq_conv(batch)

        batch = self.upblock1(batch)
        batch = F.relu(batch)
        batch = self.upblock2(batch)
        batch = F.relu(batch)
        batch = self._conv_1(batch)

        batch = self._residual_stack(batch)

        batch = self._conv_trans_1(batch)
        batch = F.relu(batch)

        return self._conv_trans_2(batch)




class UnetEncoder(nn.Module):
    """Encoder of the Unet network.
    
    Args:
        in_channels (int): Number of input channels.
        base_width (int): Base dimensionality of the layers of the autoencoder.
    """

    def __init__(self, in_channels: int, base_width: int):
        super().__init__()
        norm_layer = nn.InstanceNorm2d
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            norm_layer(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            norm_layer(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            norm_layer(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            norm_layer(base_width * 2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True))

    def forward(self, batch: Tensor):
        """Encodes batch of images into a latent representation.
        
        Args:
            inputs (Tensor): Quantized features.
        
        Returns:
            Latent representations of the input batch.
        """
        b1 = self.block1(batch)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        return b1, b2, b3, b4


class UnetDecoder(nn.Module):
    """Decoder of the Unet network.
    
    Args:
        base_width (int): Base dimensionality of the layers of the autoencoder.
        out_channels (int): Number of output channels.
    """

    def __init__(self, base_width: int, out_channels: int = 1):
        super().__init__()
        norm_layer = nn.InstanceNorm2d
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
                                 norm_layer(base_width * 4),
                                 nn.ReLU(inplace=True))
        # cat with base*4
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * (4 + 4), base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
                                 norm_layer(base_width * 2),
                                 nn.ReLU(inplace=True))
        # cat with base*2
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * (2 + 2), base_width * 2, kernel_size=3, padding=1),
            norm_layer(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            norm_layer(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
                                 norm_layer(base_width),
                                 nn.ReLU(inplace=True))
        # cat with base*1
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * (1 + 1), base_width, kernel_size=3, padding=1),
            norm_layer(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            norm_layer(base_width),
            nn.ReLU(inplace=True)
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, b1: Tensor, b2: Tensor, b3: Tensor, b4: Tensor):
        """Decodes latent represnetations into an image.
        
        Args:
            b1 (Tensor): First (top level) quantized feature map.
            b2 (Tensor): Second quantized feature map.
            b3 (Tensor): Third quantized feature map.
            b4 (Tensor): Fourth (bottom level) quantized feature map.
        
        Returns:
            Reconstructed image.
        """
        up1 = self.up1(b4)
        cat1 = torch.cat((up1, b3), dim=1)
        db1 = self.db1(cat1)

        up2 = self.up2(db1)
        cat2 = torch.cat((up2, b2), dim=1)
        db2 = self.db2(cat2)

        up3 = self.up3(db2)
        cat3 = torch.cat((up3, b1), dim=1)
        db3 = self.db3(cat3)

        out = self.fin_out(db3)
        return out



class UnetModel(nn.Module):
    """Autoencoder model that reconstructs the input image.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        base_width (int): Base dimensionality of the layers of the autoencoder.
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 64, base_width: int = 64):
        super().__init__()
        self.encoder = UnetEncoder(in_channels, base_width)
        self.decoder = UnetDecoder(base_width, out_channels=out_channels)

    def forward(self, batch: Tensor):
        """Reconstructs an input batch of images.
        
        Args:
            batch (Tensor): Batch of input images.
        
        Returns:
            Reconstructed images.
        """
        b1, b2, b3, b4 = self.encoder(batch)
        output = self.decoder(b1, b2, b3, b4)
        return output

class AnomalyDetectionModule(nn.Module):
    """Module that detects the preence of an anomaly by comparing two images reconstructed by
    the object specific decoder and the general object decoder.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        base_width (int): Base dimensionality of the layers of the autoencoder.
    """

    def __init__(self, in_channels: int = 6, out_channels: int = 2, base_width: int= 64):
        super(AnomalyDetectionModule, self).__init__()
        self.unet = UnetModel(in_channels, out_channels, base_width)

    def forward(self, batch_real: Tensor, batch_anomaly: Tensor):
        """Computes the anomaly map over corresponding real and anomalous images.
        
        Args:
            batch_real (Tensor): Batch of real, non defective images.
            batch_anomaly (Tensor): Batch of potentially anomalous images.
            
        Returns:
            The anomaly segmentation map.
        """
        img_x = torch.cat((batch_real, batch_anomaly),dim=1)
        x = self.unet(img_x)
        return x


class UpsamplingModule(nn.Module):
    """Module that upsamples the generated anomaly mask to full resolution.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        base_width (int): Base dimensionality of the layers of the autoencoder.
    """

    def __init__(self, in_channels: int = 8, out_channels: int = 2, base_width: int = 64):
        super(UpsamplingModule, self).__init__()
        self.unet = UnetModel(in_channels, out_channels, base_width)
        
    def forward(self, batch_real: Tensor, batch_anomaly: Tensor, batch_segmentation_map: Tensor):
        """Computes upsampled segmentation maps.
        
        Args:
            batch_real (Tensor): Batch of real, non defective images.
            batch_anomaly (Tensor): Batch of potentially anomalous images.
            batch_segmentation_map (Tensor): Batch of anomaly segmentation maps.
            
        Returns:
            Upsampled anomaly segmentation maps.
        """
        img_x = torch.cat((batch_real, batch_anomaly, batch_segmentation_map),dim=1)
        x = self.unet(img_x)
        return x
