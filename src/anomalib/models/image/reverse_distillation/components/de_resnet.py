"""PyTorch model defining the decoder network for Reverse Distillation.

This module implements the decoder network used in the Reverse Distillation model
architecture. The decoder reconstructs features from the bottleneck representation
back to the original feature space.

The module contains:
- Decoder block implementations using transposed convolutions
- Helper functions for creating decoder layers
- Full decoder network architecture

Example:
    >>> from anomalib.models.image.reverse_distillation.components.de_resnet import (
    ...     get_decoder
    ... )
    >>> decoder = get_decoder()
    >>> features = torch.randn(32, 512, 28, 28)
    >>> reconstructed = decoder(features)

See Also:
    - :class:`anomalib.models.image.reverse_distillation.torch_model.ReverseDistillationModel`:
        Main model implementation using this decoder
    - :class:`anomalib.models.image.reverse_distillation.components.DecoderBasicBlock`:
        Basic building block for the decoder network
"""

# Original Code
# Copyright (c) 2022 hq-deng
# https://github.com/hq-deng/RD4AD
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import torch
from torch import nn
from torchvision.models.resnet import conv1x1, conv3x3


class DecoderBasicBlock(nn.Module):
    """Basic block for decoder ResNet architecture.

    This module implements a basic decoder block used in the decoder network. It performs
    upsampling and feature reconstruction through transposed convolutions and skip
    connections.

    The block consists of:
    1. Optional upsampling via transposed convolution when ``stride=2``
    2. Two convolutional layers with batch normalization and ReLU activation
    3. Skip connection that adds input to output features

    Args:
        inplanes (int): Number of input channels
        planes (int): Number of output channels
        stride (int, optional): Stride for convolution and transposed convolution.
            When ``stride=2``, upsampling is performed. Defaults to ``1``.
        upsample (nn.Module | None, optional): Module used for upsampling the
            identity branch. Defaults to ``None``.
        groups (int, optional): Number of blocked connections from input to output
            channels. Must be ``1``. Defaults to ``1``.
        base_width (int, optional): Width of intermediate conv layers. Must be
            ``64``. Defaults to ``64``.
        dilation (int, optional): Dilation rate for convolutions. Must be ``1``.
            Defaults to ``1``.
        norm_layer (Callable[..., nn.Module] | None, optional): Normalization layer
            to use. Defaults to ``None`` which uses ``BatchNorm2d``.

    Raises:
        ValueError: If ``groups != 1`` or ``base_width != 64``
        NotImplementedError: If ``dilation > 1``

    Example:
        >>> block = DecoderBasicBlock(64, 128, stride=2)
        >>> x = torch.randn(1, 64, 32, 32)
        >>> output = block(x)  # Shape: (1, 128, 64, 64)

    Notes:
        - When ``stride=2``, the first conv is replaced with transposed conv for
          upsampling
        - The block maintains the same architectural pattern as ResNet's BasicBlock
          but in reverse
        - Skip connections help preserve spatial information during reconstruction
    """

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            msg = "BasicBlock only supports groups=1 and base_width=64"
            raise ValueError(msg)
        if dilation > 1:
            msg = "Dilation > 1 not supported in BasicBlock"
            raise NotImplementedError(msg)
        # Both self.conv1 and self.downsample layers downsample the input when stride != 2
        if stride == 2:
            self.conv1 = nn.ConvTranspose2d(
                inplanes,
                planes,
                kernel_size=2,
                stride=stride,
                groups=groups,
                bias=False,
                dilation=dilation,
            )
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder basic block.

        Args:
            batch (torch.Tensor): Input tensor of shape ``(B, C, H, W)``

        Returns:
            torch.Tensor: Output tensor of shape ``(B, C', H', W')``, where C' is
                determined by ``planes`` and H', W' depend on ``stride``
        """
        identity = batch

        out = self.conv1(batch)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(batch)

        out += identity
        return self.relu(out)


class DecoderBottleneck(nn.Module):
    """Bottleneck block for the decoder network.

    This module implements a bottleneck block used in the decoder part of the Reverse
    Distillation model. It performs upsampling and feature reconstruction through a series of
    convolutional layers.

    The block consists of three convolution layers:
    1. 1x1 conv to adjust channels
    2. 3x3 conv (or transpose conv) for processing
    3. 1x1 conv to expand channels

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of intermediate channels (will be expanded by ``expansion``).
        stride (int, optional): Stride for convolution and transpose convolution layers.
            Defaults to ``1``.
        upsample (nn.Module | None, optional): Module used for upsampling the residual branch.
            Defaults to ``None``.
        groups (int, optional): Number of blocked connections from input to output channels.
            Defaults to ``1``.
        base_width (int, optional): Base width for the conv layers.
            Defaults to ``64``.
        dilation (int, optional): Dilation rate for conv layers.
            Defaults to ``1``.
        norm_layer (Callable[..., nn.Module] | None, optional): Normalization layer to use.
            Defaults to ``None`` which will use ``nn.BatchNorm2d``.

    Attributes:
        expansion (int): Channel expansion factor (4 for bottleneck blocks).

    Example:
        >>> import torch
        >>> from anomalib.models.image.reverse_distillation.components.de_resnet import (
        ...     DecoderBottleneck
        ... )
        >>> layer = DecoderBottleneck(256, 64)
        >>> x = torch.randn(32, 256, 28, 28)
        >>> output = layer(x)
        >>> output.shape
        torch.Size([32, 256, 28, 28])

    Notes:
        - When ``stride=2``, the middle conv layer becomes a transpose conv for upsampling
        - The actual output channels will be ``planes * expansion``
    """

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 2
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if stride == 2:
            self.conv2 = nn.ConvTranspose2d(
                width,
                width,
                kernel_size=2,
                stride=stride,
                groups=groups,
                bias=False,
                dilation=dilation,
            )
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder bottleneck block.

        Args:
            batch (torch.Tensor): Input tensor of shape ``(B, C, H, W)``

        Returns:
            torch.Tensor: Output tensor of shape ``(B, C', H', W')``, where ``C'`` is
                ``planes * expansion`` and ``H'``, ``W'`` depend on ``stride``
        """
        identity = batch

        out = self.conv1(batch)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(batch)

        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    """Decoder ResNet model for feature reconstruction.

    This module implements a decoder version of the ResNet architecture, which
    reconstructs features from a bottleneck representation back to higher
    dimensional feature spaces.

    The decoder consists of multiple layers that progressively upsample and
    reconstruct features through transposed convolutions and skip connections.

    Args:
        block (Type[DecoderBasicBlock | DecoderBottleneck]): Type of decoder block
            to use in each layer. Can be either ``DecoderBasicBlock`` or
            ``DecoderBottleneck``.
        layers (list[int]): List specifying number of blocks in each decoder
            layer.
        zero_init_residual (bool, optional): If ``True``, initializes the last
            batch norm in each layer to zero. This improves model performance by
            0.2~0.3% according to https://arxiv.org/abs/1706.02677.
            Defaults to ``False``.
        groups (int, optional): Number of blocked connections from input channels
            to output channels per layer. Defaults to ``1``.
        width_per_group (int, optional): Number of channels in each intermediate
            convolution layer. Defaults to ``64``.
        norm_layer (Callable[..., nn.Module] | None, optional): Normalization
            layer to use. If ``None``, uses ``BatchNorm2d``. Defaults to ``None``.

    Example:
        >>> from anomalib.models.image.reverse_distillation.components import (
        ...     DecoderBasicBlock,
        ...     ResNet
        ... )
        >>> model = ResNet(
        ...     block=DecoderBasicBlock,
        ...     layers=[2, 2, 2, 2]
        ... )
        >>> x = torch.randn(1, 512, 8, 8)
        >>> features = model(x)  # Returns list of features at different scales

    Notes:
        - The decoder reverses the typical ResNet architecture, starting from a
          bottleneck and expanding to larger feature maps
        - Features are returned at multiple scales for multi-scale reconstruction
        - The implementation follows the original ResNet paper but in reverse
          for decoding

    See Also:
        - :class:`DecoderBasicBlock`: Basic building block for decoder layers
        - :class:`DecoderBottleneck`: Bottleneck building block for deeper
          decoder architectures
    """

    def __init__(
        self,
        block: type[DecoderBasicBlock | DecoderBottleneck],
        layers: list[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 512 * block.expansion
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d | nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, DecoderBottleneck):
                    nn.init.constant_(module.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(module, DecoderBasicBlock):
                    nn.init.constant_(module.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: type[DecoderBasicBlock | DecoderBottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=2,
                    stride=stride,
                    groups=self.groups,
                    bias=False,
                    dilation=self.dilation,
                ),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, upsample, self.groups, self.base_width, previous_dilation, norm_layer),
        )
        self.inplanes = planes * block.expansion
        layers.extend(
            [
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
                for _ in range(1, blocks)
            ],
        )

        return nn.Sequential(*layers)

    def forward(self, batch: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass through the decoder ResNet.

        Progressively reconstructs features through multiple decoder layers,
        returning features at different scales.

        Args:
            batch (torch.Tensor): Input tensor of shape ``(B, C, H, W)`` where:
                - ``B`` is batch size
                - ``C`` is number of input channels (512 * block.expansion)
                - ``H`` and ``W`` are spatial dimensions

        Returns:
            list[torch.Tensor]: List of feature tensors at different scales:
                - ``feature_c``: ``(B, 64, H*8, W*8)``
                - ``feature_b``: ``(B, 128, H*4, W*4)``
                - ``feature_a``: ``(B, 256, H*2, W*2)``

        Example:
            >>> model = ResNet(DecoderBasicBlock, [2, 2, 2])
            >>> x = torch.randn(1, 512, 8, 8)
            >>> features = model(x)
            >>> [f.shape for f in features]
            [(1, 64, 64, 64), (1, 128, 32, 32), (1, 256, 16, 16)]
        """
        feature_a = self.layer1(batch)  # 512*8*8->256*16*16
        feature_b = self.layer2(feature_a)  # 256*16*16->128*32*32
        feature_c = self.layer3(feature_b)  # 128*32*32->64*64*64

        return [feature_c, feature_b, feature_a]


def _resnet(block: type[DecoderBasicBlock | DecoderBottleneck], layers: list[int], **kwargs) -> ResNet:
    return ResNet(block, layers, **kwargs)


def de_resnet18() -> ResNet:
    """ResNet-18 model."""
    return _resnet(DecoderBasicBlock, [2, 2, 2, 2])


def de_resnet34() -> ResNet:
    """ResNet-34 model."""
    return _resnet(DecoderBasicBlock, [3, 4, 6, 3])


def de_resnet50() -> ResNet:
    """ResNet-50 model."""
    return _resnet(DecoderBottleneck, [3, 4, 6, 3])


def de_resnet101() -> ResNet:
    """ResNet-101 model."""
    return _resnet(DecoderBottleneck, [3, 4, 23, 3])


def de_resnet152() -> ResNet:
    """ResNet-152 model."""
    return _resnet(DecoderBottleneck, [3, 8, 36, 3])


def de_resnext50_32x4d() -> ResNet:
    """ResNeXt-50 32x4d model."""
    return _resnet(DecoderBottleneck, [3, 4, 6, 3], groups=32, width_per_group=4)


def de_resnext101_32x8d() -> ResNet:
    """ResNeXt-101 32x8d model."""
    return _resnet(DecoderBottleneck, [3, 4, 23, 3], groups=32, width_per_group=8)


def de_wide_resnet50_2() -> ResNet:
    """Wide ResNet-50-2 model."""
    return _resnet(DecoderBottleneck, [3, 4, 6, 3], width_per_group=128)


def de_wide_resnet101_2() -> ResNet:
    """Wide ResNet-101-2 model."""
    return _resnet(DecoderBottleneck, [3, 4, 23, 3], width_per_group=128)


def get_decoder(name: str) -> ResNet:
    """Get decoder model based on the name of the backbone.

    Args:
        name (str): Name of the backbone.

    Returns:
        ResNet: Decoder ResNet architecture.
    """
    if name in {
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "wide_resnet50_2",
        "wide_resnet101_2",
    }:
        decoder = globals()[f"de_{name}"]
    else:
        msg = f"Decoder with architecture {name} not supported"
        raise ValueError(msg)
    return decoder()
