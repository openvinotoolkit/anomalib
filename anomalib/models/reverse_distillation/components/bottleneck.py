"""Torch model defining the bottleneck layer."""

# Original Code
# Copyright (c) 2022 hq-deng
# https://github.com/hq-deng/RD4AD
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn
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

    Args:
        block (Bottleneck): Expansion value is extracted from this block.
        layers (int): Numbers of OCE layers to create after multiscale feature fusion.
        groups (int, optional): Number of blocked connections from input channels to output channels.
            Defaults to 1.
        width_per_group (int, optional): Number of layers in each intermediate convolution layer. Defaults to 64.
        norm_layer (Callable[..., nn.Module] | None, optional): Batch norm layer to use. Defaults to None.
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

        # This is present in the paper but not in the original code. With some initial experiments, removing this leads
        # to better results
        # self.conv4 = conv1x1(256 * block.expansion * 3, 256 * block.expansion * 3, 1)  # x3 as we concatenate 3 layers
        # self.bn4 = norm_layer(256 * block.expansion * 3)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
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
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, features: list[Tensor]) -> Tensor:
        """Forward-pass of Bottleneck layer.

        Args:
            features (list[Tensor]): List of features extracted from the encoder.

        Returns:
            Tensor: Output of the bottleneck layer
        """
        # Always assumes that features has length of 3
        feature0 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(features[0]))))))
        feature1 = self.relu(self.bn3(self.conv3(features[1])))
        feature_cat = torch.cat([feature0, feature1, features[2]], 1)
        output = self.bn_layer(feature_cat)
        # output = self.bn_layer(self.bn4(self.conv4(feature_cat)))

        return output.contiguous()


def get_bottleneck_layer(backbone: str, **kwargs) -> OCBE:
    """Get appropriate bottleneck layer based on the name of the backbone.

    Args:
        backbone (str): Name of the backbone.

    Returns:
        Bottleneck_layer: One-Class Bottleneck Embedding module.
    """
    if backbone in ("resnet18", "resnet34"):
        ocbe = OCBE(BasicBlock, 2, **kwargs)
    else:
        ocbe = OCBE(Bottleneck, 3, **kwargs)

    return ocbe
