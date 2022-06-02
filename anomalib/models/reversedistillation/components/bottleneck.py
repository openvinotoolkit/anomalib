"""Torch model defining the bottleneck layer."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# Code adapted from https://github.com/hq-deng/RD4AD


from typing import Callable, List, Optional, Type

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import Bottleneck


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
        norm_layer (Optional[Callable[..., nn.Module]], optional): Batch norm layer to use. Defaults to None.
    """

    def __init__(
        self,
        block: Type[Bottleneck],
        layers: int,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
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

        self.conv4 = conv1x1(1024 * block.expansion, 512 * block.expansion, 1)
        self.bn4 = norm_layer(512 * block.expansion)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[Bottleneck],
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

    def forward(self, features: List[Tensor]) -> Tensor:
        """Forward-pass of Bottleneck layer.

        Args:
            features (List[Tensor]): List of features extracted from the encoder.

        Returns:
            Tensor: Output of the bottleneck layer
        """
        # Always assumes that features has length of 3
        l1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(features[0]))))))
        l2 = self.relu(self.bn3(self.conv3(features[1])))
        feature = torch.cat([l1, l2, features[2]], 1)
        output = self.bn_layer(feature)

        return output.contiguous()


def get_bottleneck_layer(backbone: str, **kwargs) -> OCBE:
    """Get appropriate bottleneck layer based on the name of the backbone.

    Args:
        backbone (str): Name of the backbone.

    Returns:
        Bottleneck_layer: One-Class Bottleneck Embedding module.
    """
    if backbone == "resnet18":
        return OCBE(Bottleneck, 2, **kwargs)
    else:
        return OCBE(Bottleneck, 3, **kwargs)
