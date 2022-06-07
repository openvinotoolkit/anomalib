"""Torch model defining the decoder."""

# Original Code
# Copyright (c) 2022 hq-deng
# https://github.com/hq-deng/RD4AD
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, List, Optional, Type, Union

from torch import Tensor, nn
from torchvision.models.resnet import conv1x1, conv3x3


class DecoderBasicBlock(nn.Module):
    """Basic block for decoder ResNet architecture.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride for convolution and de-convolution layers. Defaults to 1.
        upsample (Optional[nn.Module], optional): Module used for upsampling output. Defaults to None.
        groups (int, optional): Number of blocked connections from input channels to output channels.
            Defaults to 1.
        base_width (int, optional): Number of layers in each intermediate convolution layer. Defaults to 64.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        norm_layer (Optional[Callable[..., nn.Module]], optional): Batch norm layer to use.Defaults to None.

    Raises:
        ValueError: If groups are not equal to 1 and base width is not 64.
        NotImplementedError: If dilation is greater than 1.
    """

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 2
        if stride == 2:
            self.conv1 = nn.ConvTranspose2d(
                inplanes, planes, kernel_size=2, stride=stride, groups=groups, bias=False, dilation=dilation
            )
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, batch: Tensor) -> Tensor:
        """Forward-pass of de-resnet block."""
        identity = batch

        out = self.conv1(batch)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(batch)

        out += identity
        out = self.relu(out)

        return out


class DecoderBottleneck(nn.Module):
    """Bottleneck for Decoder.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride for convolution and de-convolution layers. Defaults to 1.
        upsample (Optional[nn.Module], optional): Module used for upsampling output. Defaults to None.
        groups (int, optional): Number of blocked connections from input channels to output channels.
            Defaults to 1.
        base_width (int, optional): Number of layers in each intermediate convolution layer. Defaults to 64.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        norm_layer (Optional[Callable[..., nn.Module]], optional): Batch norm layer to use.Defaults to None.
    """

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
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
                width, width, kernel_size=2, stride=stride, groups=groups, bias=False, dilation=dilation
            )
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, batch: Tensor) -> Tensor:
        """Forward-pass of de-resnet bottleneck block."""
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
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet model for decoder.

    Args:
        block (Type[Union[DecoderBasicBlock, DecoderBottleneck]]): Type of block to use in a layer.
        layers (List[int]): List to specify number for blocks per layer.
        zero_init_residual (bool, optional): If true, initializes the last batch norm in each layer to zero.
            Defaults to False.
        groups (int, optional): Number of blocked connections per layer from input channels to output channels.
            Defaults to 1.
        width_per_group (int, optional): Number of layers in each intermediate convolution layer.. Defaults to 64.
        norm_layer (Optional[Callable[..., nn.Module]], optional): Batch norm layer to use. Defaults to None.
    """

    def __init__(
        self,
        block: Type[Union[DecoderBasicBlock, DecoderBottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
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
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
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
        block: Type[Union[DecoderBasicBlock, DecoderBottleneck]],
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
            block(self.inplanes, planes, stride, upsample, self.groups, self.base_width, previous_dilation, norm_layer)
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

    def forward(self, batch: Tensor) -> List[Tensor]:
        """Forward pass for Decoder ResNet. Returns list of features."""
        feature_a = self.layer1(batch)  # 512*8*8->256*16*16
        feature_b = self.layer2(feature_a)  # 256*16*16->128*32*32
        feature_c = self.layer3(feature_b)  # 128*32*32->64*64*64

        return [feature_c, feature_b, feature_a]


def _resnet(block: Type[Union[DecoderBasicBlock, DecoderBottleneck]], layers: List[int], **kwargs: Any) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


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
    if name in (
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "wide_resnet50_2",
        "wide_resnet101_2",
    ):
        decoder = globals()[f"de_{name}"]
    else:
        raise ValueError(f"Decoder with architecture {name} not supported")
    return decoder()
