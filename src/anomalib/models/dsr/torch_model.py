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

    def forward(self, batch : Tensor, quantization : function | object) -> tuple[Tensor, Tensor]:
        """Generate the quantized anomaly-free representation of an anomalous image.
        
        Args:
            batch (Tensor): Batch of input images.
            quantization (function | object): Quantization function.
        
        Returns:
            Reconstructed batch of non-quantized features and corresponding quantized features.
        """
        batch = self.unet(batch)
        quantized_b = quantization(batch)
        return batch, quantized_b

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

    def forward(self, batch: Tensor) -> Tensor:
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

    def forward(self, batch: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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

    def forward(self, b1: Tensor, b2: Tensor, b3: Tensor) -> Tensor:
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

    def forward(self, batch: Tensor) -> Tensor:
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

    def forward(self, batch: Tensor) -> Tensor:
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

    def forward(self, inputs: Tensor) -> Tensor:
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

    def forward(self, batch: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
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

    def forward(self, b1: Tensor, b2: Tensor, b3: Tensor, b4: Tensor) -> Tensor:
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

    def forward(self, batch: Tensor) -> Tensor:
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

    def forward(self, batch_real: Tensor, batch_anomaly: Tensor) -> Tensor:
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
        
    def forward(self, batch_real: Tensor, batch_anomaly: Tensor, batch_segmentation_map: Tensor) -> Tensor:
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
    

class VectorQuantizer(nn.Module):
    """Module that quantizes a given feature map using learned quantization codebooks.
    
    Args:
        num_embeddings (int): Size of embedding codebook.
        embedding_dim (int): Dimension of embeddings.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Source for the VectorQuantizer module: https://github.com/zalandoresearch/pytorch-vq-vae
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()

        # necessary to correctly load the checkpoint file
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

    def forward(self, inputs: Tensor) -> Tensor:
        """Calculates quantized feature map.
        
        Args:
            inputs (Tensor): Non-quantized feature maps.
        
        Returns:
            Quantized feature maps.
        """
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        quantized = inputs + (quantized - inputs).detach()

        # convert quantized from BHWC -> BCHW
        return quantized.permute(0, 3, 1, 2).contiguous()


class EncoderBot(nn.Module):
    """Encoder module for bottom quantized feature maps.
    
    Args:
        in_channels (int): Number of input channels.
        num_hiddens (int): Number of hidden channels.
        num_residual_layers (int): Number of residual layers in residual stacks.
        num_residual_hiddens (int): Number of channels in residual layers.
    """
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int):
        super(EncoderBot, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, batch: Tensor) -> Tensor:
        """Encode inputs to be quantized into the bottom feature map.
        
        Args:
            batch (Tensor): Batch of input images.
        
        Returns:
            Encoded feature maps.
        """
        x = self._conv_1(batch)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


class EncoderTop(nn.Module):
    """Encoder module for top quantized feature maps.
    
    Args:
        in_channels (int): Number of input channels.
        num_hiddens (int): Number of hidden channels.
        num_residual_layers (int): Number of residual layers in residual stacks.
        num_residual_hiddens (int): Number of channels in residual layers.
    """
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int):
        super(EncoderTop, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, batch: Tensor) -> Tensor:
        """Encode inputs to be quantized into the top feature map.
        
        Args:
            batch (Tensor): Batch of input images.
        
        Returns:
            Encoded feature maps.
        """
        x = self._conv_1(batch)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._residual_stack(x)
        return x


class DecoderBot(nn.Module):
    """Decoder module for bottom quantized feature maps.
    
    Args:
        in_channels (int): Number of input channels.
        num_hiddens (int): Number of hidden channels.
        num_residual_layers (int): Number of residual layers in residual stack.
        num_residual_hiddens (int): Number of channels in residual layers.
    """
    def __init__(self, in_channels: int, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int):
        super(DecoderBot, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
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

    def forward(self, batch: Tensor) -> Tensor:
        """Decode bottom feature maps into top feature maps.
        
        Args:
            batch (Tensor): Batch of input images.
        
        Returns:
            Decoded top feature maps.
        """
        x = self._conv_1(batch)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)

class DiscreteLatentModel(nn.Module):
    """Autoencoder quantized model that encodes the input images into quantized feature maps and generates
    a reconstructed image using the quantized feature maps.
    
    Args:
        num_hiddens (int): Number of hidden channels.
        num_residual_layers (int): Number of residual layers in residual stacks.
        num_residual_hiddens (int): Number of channels in residual layers.
        num_embeddings (int): Size of embedding dictionary.
        embedding_dim (int): Dimension of embeddings.
    """
    def __init__(self, num_hiddens: int, num_residual_layers: int, num_residual_hiddens: int, num_embeddings: int, embedding_dim: int):
        super(DiscreteLatentModel, self).__init__()

        self._encoder_t = EncoderTop(num_hiddens, num_hiddens,
                                     num_residual_layers,
                                     num_residual_hiddens)

        self._encoder_b = EncoderBot(3, num_hiddens,
                                     num_residual_layers,
                                     num_residual_hiddens)

        self._pre_vq_conv_bot = nn.Conv2d(in_channels=num_hiddens + embedding_dim,
                                          out_channels=embedding_dim,
                                          kernel_size=1,
                                          stride=1)

        self._pre_vq_conv_top = nn.Conv2d(in_channels=num_hiddens,
                                          out_channels=embedding_dim,
                                          kernel_size=1,
                                          stride=1)

        self._vq_vae_top = VectorQuantizer(num_embeddings, embedding_dim)

        self._vq_vae_bot = VectorQuantizer(num_embeddings, embedding_dim)

        self._decoder_b = DecoderBot(embedding_dim*2,
                                     num_hiddens,
                                     num_residual_layers,
                                     num_residual_hiddens)


        self.upsample_t = nn.ConvTranspose2d(
            embedding_dim, embedding_dim, 4, stride=2, padding=1
        )


    def forward(self, batch: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Generates quantized feature maps of batch of input images as well as their
        reconstruction based on its auqntized feature maps.
        
        Args:
            batch (Tensor): Batch of input images.
        
        Returns:
            Tuple of reconstructed images, quantized top feature maps, and quantized
            bottom feature maps.
        """
        #Encoder Hi
        enc_b = self._encoder_b(batch)

        #Encoder Lo -- F_Lo
        enc_t = self._encoder_t(enc_b)
        zt = self._pre_vq_conv_top(enc_t)

        # Quantize F_Lo with K_Lo
        quantized_t = self._vq_vae_top(zt)
        # Upsample Q_Lo
        up_quantized_t = self.upsample_t(quantized_t)

        # Concatenate and transform the output of Encoder_Hi and upsampled Q_lo -- F_Hi
        feat = torch.cat((enc_b, up_quantized_t), dim=1)
        zb = self._pre_vq_conv_bot(feat)

        # Quantize F_Hi with K_Hi
        quantized_b = self._vq_vae_bot(zb)

        # Concatenate Q_Hi and Q_Lo and input it into the General appearance decoder
        quant_join = torch.cat((up_quantized_t, quantized_b), dim=1)
        recon_fin = self._decoder_b(quant_join)

        return recon_fin, quantized_t, quantized_b
