"""SSPCAB: Self-Supervised Predictive Convolutional Attention Block.

This module implements the SSPCAB architecture from the paper:
"SSPCAB: Self-Supervised Predictive Convolutional Attention Block for
Reconstruction-Based Anomaly Detection"
(https://arxiv.org/abs/2111.09099)

The SSPCAB combines masked convolutions with channel attention to learn
spatial-spectral feature representations for anomaly detection.

Example:
    >>> import torch
    >>> from anomalib.models.components.layers import SSPCAB
    >>> # Create SSPCAB layer
    >>> sspcab = SSPCAB(in_channels=64)
    >>> # Apply attention to input tensor
    >>> x = torch.randn(1, 64, 32, 32)
    >>> output = sspcab(x)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


class AttentionModule(nn.Module):
    """Squeeze and excitation block that acts as the attention module in SSPCAB.

    This module applies channel attention through global average pooling followed
    by two fully connected layers with non-linearities.

    Args:
        in_channels (int): Number of input channels.
        reduction_ratio (int, optional): Reduction ratio for the intermediate
            layer. The intermediate layer will have ``in_channels //
            reduction_ratio`` channels. Defaults to 8.

    Example:
        >>> import torch
        >>> from anomalib.models.components.layers.sspcab import AttentionModule
        >>> attention = AttentionModule(in_channels=64)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> output = attention(x)
        >>> output.shape
        torch.Size([1, 64, 32, 32])
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 8) -> None:
        super().__init__()

        out_channels = in_channels // reduction_ratio
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, in_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the attention module.

        Args:
            inputs (torch.Tensor): Input tensor of shape
                ``(batch_size, channels, height, width)``.

        Returns:
            torch.Tensor: Attended output tensor of same shape as input.
        """
        # reduce feature map to 1d vector through global average pooling
        avg_pooled = inputs.mean(dim=(2, 3))

        # squeeze and excite
        act = self.fc1(avg_pooled)
        act = F.relu(act)
        act = self.fc2(act)
        act = F.sigmoid(act)

        # multiply with input
        return inputs * act.view(act.shape[0], act.shape[1], 1, 1)


class SSPCAB(nn.Module):
    """Self-Supervised Predictive Convolutional Attention Block.

    This module combines masked convolutions with channel attention to capture
    spatial and channel dependencies in the feature maps.

    Args:
        in_channels (int): Number of input channels.
        kernel_size (int, optional): Size of the receptive fields of the masked
            convolution kernel. Defaults to 1.
        dilation (int, optional): Dilation factor of the masked convolution
            kernel. Defaults to 1.
        reduction_ratio (int, optional): Reduction ratio of the attention module.
            Defaults to 8.

    Example:
        >>> import torch
        >>> from anomalib.models.components.layers import SSPCAB
        >>> sspcab = SSPCAB(in_channels=64, kernel_size=3)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> output = sspcab(x)
        >>> output.shape
        torch.Size([1, 64, 32, 32])
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 1,
        dilation: int = 1,
        reduction_ratio: int = 8,
    ) -> None:
        super().__init__()

        self.pad = kernel_size + dilation
        self.crop = kernel_size + 2 * dilation + 1

        self.masked_conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
        )
        self.masked_conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
        )
        self.masked_conv3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
        )
        self.masked_conv4 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
        )

        self.attention_module = AttentionModule(
            in_channels=in_channels,
            reduction_ratio=reduction_ratio,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the SSPCAB block.

        Args:
            inputs (torch.Tensor): Input tensor of shape
                ``(batch_size, channels, height, width)``.

        Returns:
            torch.Tensor: Output tensor of same shape as input.
        """
        # compute masked convolution
        padded = F.pad(inputs, (self.pad,) * 4)
        masked_out = torch.zeros_like(inputs)
        masked_out += self.masked_conv1(padded[..., : -self.crop, : -self.crop])
        masked_out += self.masked_conv2(padded[..., : -self.crop, self.crop :])
        masked_out += self.masked_conv3(padded[..., self.crop :, : -self.crop])
        masked_out += self.masked_conv4(padded[..., self.crop :, self.crop :])

        # apply channel attention module
        return self.attention_module(masked_out)
