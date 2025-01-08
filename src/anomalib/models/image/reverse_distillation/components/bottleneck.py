"""PyTorch model defining the bottleneck layer for Reverse Distillation.

This module implements the bottleneck layer used in the Reverse Distillation model
architecture. The bottleneck layer compresses features into a lower dimensional
space while preserving important information for anomaly detection.

The module contains:
- Bottleneck layer implementation using convolutional blocks
- Helper functions for creating 3x3 and 1x1 convolutions
- One-Class Bottleneck Embedding (OCBE) module for feature compression

Example:
    >>> from anomalib.models.image.reverse_distillation.components.bottleneck import (
    ...     get_bottleneck_layer
    ... )
    >>> bottleneck = get_bottleneck_layer()
    >>> features = torch.randn(32, 512, 28, 28)
    >>> compressed = bottleneck(features)

See Also:
    - :class:`anomalib.models.image.reverse_distillation.torch_model.ReverseDistillationModel`:
        Main model implementation using this bottleneck layer
    - :class:`anomalib.models.image.reverse_distillation.components.OCBE`:
        One-Class Bottleneck Embedding module
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
from torchvision.models.resnet import BasicBlock, Bottleneck


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class OCBE(nn.Module):
    """One-Class Bottleneck Embedding module.

    This module implements a bottleneck layer that compresses multi-scale features into a
    compact representation. It consists of:

    1. Multiple convolutional layers to process features at different scales
    2. Feature fusion through concatenation
    3. Final bottleneck compression through residual blocks

    The module takes features from multiple scales of an encoder network and outputs a
    compressed bottleneck representation.

    Args:
        block (Bottleneck | BasicBlock): Block type that determines expansion factor.
            Can be either ``Bottleneck`` or ``BasicBlock``.
        layers (int): Number of OCE layers to create after multi-scale feature fusion.
        groups (int, optional): Number of blocked connections from input channels to
            output channels. Defaults to ``1``.
        width_per_group (int, optional): Number of channels in each intermediate
            convolution layer. Defaults to ``64``.
        norm_layer (Callable[..., nn.Module] | None, optional): Normalization layer to
            use. If ``None``, uses ``BatchNorm2d``. Defaults to ``None``.

    Example:
        >>> import torch
        >>> from torchvision.models.resnet import Bottleneck
        >>> from anomalib.models.image.reverse_distillation.components import OCBE
        >>> model = OCBE(block=Bottleneck, layers=3)
        >>> # Create 3 feature maps of different scales
        >>> f1 = torch.randn(1, 256, 28, 28)  # First scale
        >>> f2 = torch.randn(1, 512, 14, 14)  # Second scale
        >>> f3 = torch.randn(1, 1024, 7, 7)   # Third scale
        >>> features = [f1, f2, f3]
        >>> output = model(features)
        >>> output.shape
        torch.Size([1, 2048, 4, 4])

    Notes:
        - The module expects exactly 3 input feature maps at different scales
        - Features are processed through conv layers before fusion
        - Final output dimensions depend on the input feature dimensions and stride
        - Initialization uses Kaiming normal for conv layers and constant for norms

    See Also:
        - :func:`get_bottleneck_layer`: Factory function to create OCBE instances
        - :class:`torchvision.models.resnet.Bottleneck`: ResNet bottleneck block
        - :class:`torchvision.models.resnet.BasicBlock`: ResNet basic block
    """

    def __init__(
        self,
        block: Bottleneck | BasicBlock,
        layers: int,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.groups = groups
        self.base_width = width_per_group
        self.inplanes = 256 * block.expansion
        self.dilation = 1
        self.bn_layer = self._make_layer(block, 512, layers, stride=2)

        self.conv1 = conv3x3(64 * block.expansion, 128 * block.expansion, 2)
        self.bn1 = norm_layer(128 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn2 = norm_layer(256 * block.expansion)
        self.conv3 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
        self.bn3 = norm_layer(256 * block.expansion)

        # self.conv4 and self.bn4 are from the original code:
        # https://github.com/hq-deng/RD4AD/blob/6554076872c65f8784f6ece8cfb39ce77e1aee12/resnet.py#L412
        self.conv4 = conv1x1(1024 * block.expansion, 512 * block.expansion, 1)
        self.bn4 = norm_layer(512 * block.expansion)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d | nn.GroupNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(
        self,
        block: type[Bottleneck | BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes * 3, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes * 3,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            ),
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

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass of the bottleneck layer.

        Processes multi-scale features through convolution layers, fuses them via
        concatenation, and applies final bottleneck compression.

        Args:
            features (list[torch.Tensor]): List of 3 feature tensors from different
                scales of the encoder network. Expected shapes:
                - features[0]: ``(B, C1, H1, W1)``
                - features[1]: ``(B, C2, H2, W2)``
                - features[2]: ``(B, C3, H3, W3)``
                where B is batch size, Ci are channel dimensions, and Hi, Wi are
                spatial dimensions.

        Returns:
            torch.Tensor: Compressed bottleneck representation with shape
                ``(B, C_out, H_out, W_out)``, where dimensions depend on the input
                feature shapes and stride values.
        """
        # Always assumes that features has length of 3
        feature0 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(features[0]))))))
        feature1 = self.relu(self.bn3(self.conv3(features[1])))
        feature_cat = torch.cat([feature0, feature1, features[2]], 1)
        output = self.bn_layer(feature_cat)

        return output.contiguous()


def get_bottleneck_layer(backbone: str, **kwargs) -> OCBE:
    """Get appropriate bottleneck layer based on the name of the backbone.

    Args:
        backbone (str): Name of the backbone.
        kwargs: Additional keyword arguments.

    Returns:
        Bottleneck_layer: One-Class Bottleneck Embedding module.
    """
    return OCBE(BasicBlock, 2, **kwargs) if backbone in {"resnet18", "resnet34"} else OCBE(Bottleneck, 3, **kwargs)
