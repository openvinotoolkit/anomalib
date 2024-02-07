"""Torch model for student, teacher and autoencoder model in EfficientAd"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import random
from enum import Enum
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import torchvision
from torchvision import transforms

logger = logging.getLogger(__name__)


def imagenet_norm_batch(x):
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to(x.device)
    x_norm = (x - mean) / std
    return x_norm


def create_pdn_model(model_type: str, out_channels: int, padding: bool = False):
    model_type_map = {
        "PDN_M_16": lambda out_channels, padding: PDN_M_16(out_channels, padding),
        "PDN_M_33": lambda out_channels, padding: PDN_M_33(out_channels, padding),
        "PDN_M_68": lambda out_channels, padding: PDN_M_68(out_channels, padding),
        "PDN_M_128": lambda out_channels, padding: PDN_M_128(out_channels, padding),
        "PDN_M_256": lambda out_channels, padding: PDN_M_256(out_channels, padding),
    }
    if model_type not in model_type_map:
        raise ValueError(f"Unknown model type {model_type}")
    return model_type_map[model_type](out_channels, padding)


def reduce_tensor_elems(tensor: torch.Tensor, m=2**24) -> torch.Tensor:
    """Flattens n-dimensional tensors,  selects m elements from it
    and returns the selected elements as tensor. It is used to select
    at most 2**24 for torch.quantile operation, as it is the maximum
    supported number of elements.
    https://github.com/pytorch/pytorch/blob/b9f81a483a7879cd3709fd26bcec5f1ee33577e6/aten/src/ATen/native/Sorting.cpp#L291

    Args:
        tensor (torch.Tensor): input tensor from which elements are selected
        m (int): number of maximum tensor elements. Default: 2**24

    Returns:
            Tensor: reduced tensor
    """
    tensor = torch.flatten(tensor)
    if len(tensor) > m:
        # select a random subset with m elements.
        perm = torch.randperm(len(tensor), device=tensor.device)
        idx = perm[:m]
        tensor = tensor[idx]
    return tensor


class EfficientAdModelSize(str, Enum):
    """Supported EfficientAd model sizes"""

    M = "medium"
    S = "small"
    PDN_M_16 = "pdn_m_16"
    PDN_M_16_DEFORM = "pdn_m_16_deform"
    PDN_M_33 = "pdn_m_33"
    PDN_M_33_DEFORM = "pdn_m_33_deform"
    PDN_M_33_DEFORM_1_2 = "pdn_m_33_deform_1_2"
    PDN_M_33_DEFORM_2_4 = "pdn_m_33_deform_2_4"
    PDN_M_33_DEFORM_4_5 = "pdn_m_33_deform_4_5"
    PDN_M_68 = "pdn_m_68"
    PDN_M_68_DEFORM = "pdn_m_68_deform"
    PDN_M_128 = "pdn_m_128"
    PDN_M_128_DEFORM = "pdn_m_128_deform"
    PDN_M_256 = "pdn_m_256"
    PDN_M_256_DEFORM = "pdn_m_256_deform"


class PDN_S(nn.Module):
    """Patch Description Network small

    Args:
        out_channels (int): number of convolution output channels
    """

    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x):
        x = imagenet_norm_batch(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x


class PDN_M(nn.Module):
    """Patch Description Network medium

    Args:
        out_channels (int): number of convolution output channels
    """

    def __init__(
        self,
        out_channels: int,
        padding: bool = False,
        from_other_network=False,
        deform_case : Optional[str] = None, # e.g. "1_2" means 1st and 2nd layer are deform conv
    ) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.from_other_network = from_other_network

        if self.from_other_network:
            if deform_case == "1_2":
                self.conv1_ = DeformableConv2d(256, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
            else:
                self.conv1_ = nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        else:
            if deform_case == "1_2":
                self.conv1 = DeformableConv2d(3, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
            else:
                self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3 * pad_mult)

        if deform_case in ["1_2", "2_4"]:
            self.conv2 = DeformableConv2d(256, 512, kernel_size=4, stride=1, padding=3 * pad_mult)
        else:
            self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3 * pad_mult)

        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0 * pad_mult)

        if deform_case in ["2_4", "4_5"]:
            self.conv4 = DeformableConv2d(512, 512, kernel_size=3, stride=1, padding=1 * pad_mult)
        else:
            self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1 * pad_mult)

        if deform_case == "4_5":
            self.conv5 = DeformableConv2d(512, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        else:
            self.conv5 = nn.Conv2d(512, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)

        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0 * pad_mult)

        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x):
        if self.from_other_network:
            x = F.relu(self.conv1_(x))
        else:
            x = imagenet_norm_batch(x)
            x = F.relu(self.conv1(x))

        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        return x


class PDN_M_16(nn.Module):
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        self.m_16_conv1 = nn.Conv2d(3, 512, kernel_size=3, stride=1, padding=0)
        self.m_16_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.m_16_conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.m_16_conv4 = nn.Conv2d(512, out_channels, kernel_size=3, stride=1, padding=0)
        self.m_16_avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = imagenet_norm_batch(x) # size: 16
        x = F.relu(self.m_16_conv1(x)) # size: 14
        x = self.m_16_avgpool1(x) # size: 7
        x = F.relu(self.m_16_conv2(x)) # size: 5
        x = F.relu(self.m_16_conv3(x)) # size: 3
        x = F.relu(self.m_16_conv4(x)) # size: 1
        return x


class PDN_M_33_Base(PDN_M):
    """Copy of the base class PDN_M"""
    pass


class PDN_M_33(PDN_M_33_Base):
    """Copy of the base class PDN_M"""
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__(out_channels, padding, deform_case=None)


class PDN_M_33_Deform(PDN_M_33_Base):
    """Copy of the base class PDN_M"""
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__(out_channels, padding, deform_case=None)


class PDN_M_33_Deform_1_2(PDN_M_33_Base):
    """Copy of the base class PDN_M"""
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__(out_channels, padding, deform_case="1_2")


class PDN_M_33_Deform_2_4(PDN_M_33_Base):
    """Copy of the base class PDN_M"""
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__(out_channels, padding, deform_case="2_4")


class PDN_M_33_Deform_4_5(PDN_M_33_Base):
    """Copy of the base class PDN_M"""
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__(out_channels, padding, deform_case="4_5")


class PDN_M_68_Base(PDN_M_33_Base):
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__(out_channels, padding, from_other_network=True)
        self.m_68_conv1 = nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=0)
        self.m_68_avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = imagenet_norm_batch(x)
        x = F.relu(self.m_68_conv1(x))
        x = self.m_68_avgpool1(x)
        x = super().forward(x)
        return x

class PDN_M_68(PDN_M_68_Base):
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__(out_channels, padding)

class PDN_M_68_Deform(PDN_M_68_Base):
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__(out_channels, padding)


class PDN_M_128_Base(PDN_M_33_Base):
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__(out_channels, padding, from_other_network=True)
        self.m_128_conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.m_128_conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2) # use padding=2
        self.m_128_avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.m_128_avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = imagenet_norm_batch(x) # size: 128
        x = F.relu(self.m_128_conv1(x)) # size: 128
        x = self.m_128_avgpool1(x) # size: 64
        x = F.relu(self.m_128_conv2(x)) # size: 66
        x = self.m_128_avgpool2(x) # size: 33
        x = super().forward(x)
        return x


class PDN_M_128(PDN_M_128_Base):
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__(out_channels, padding)


class PDN_M_128_Deform(PDN_M_128_Base):
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__(out_channels, padding)


class PDN_M_256_Base(PDN_M_33_Base):
    def __init__(self, out_channels: int, padding: bool = False, use_deform: bool = False) -> None:
        super().__init__(out_channels, padding, from_other_network=True, use_deform=use_deform)
        self.m_256_conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.m_256_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.m_256_conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2) # use padding=2
        self.m_256_avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.m_256_avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.m_256_avgpool3 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = imagenet_norm_batch(x) # size: 256
        x = F.relu(self.m_256_conv1(x)) # size: 256
        x = self.m_256_avgpool1(x) # size: 128
        x = F.relu(self.m_256_conv2(x)) # size: 128
        x = self.m_256_avgpool2(x) # size: 64
        x = F.relu(self.m_256_conv3(x)) # size: 66
        x = self.m_256_avgpool3(x) # size: 33
        x = super().forward(x)
        return x


class PDN_M_256(PDN_M_256_Base):
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__(out_channels, padding, use_deform=False)


class PDN_M_256_Deform(PDN_M_256_Base):
    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__(out_channels, padding, use_deform=True)


class Encoder(nn.Module):
    """Autoencoder Encoder model."""

    def __init__(self) -> None:
        super().__init__()
        self.enconv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)

    def forward(self, x):
        x = F.relu(self.enconv1(x))
        x = F.relu(self.enconv2(x))
        x = F.relu(self.enconv3(x))
        x = F.relu(self.enconv4(x))
        x = F.relu(self.enconv5(x))
        x = self.enconv6(x)
        return x


class Decoder(nn.Module):
    """Autoencoder Decoder model.

    Args:
        out_channels (int): number of convolution output channels
        img_size (tuple): size of input images
    """

    def __init__(self, out_channels, padding, img_size, special_model_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.img_size = img_size


        if special_model_size == EfficientAdModelSize.PDN_M_33:
            self.last_upsample = (
                int(img_size[0] / 4) if padding else int(img_size[0] / 4) - 8, # 256,551
                int(img_size[1] / 4) if padding else int(img_size[1] / 4) - 7, # 256,551
            )
        elif (
            special_model_size == EfficientAdModelSize.PDN_M_33_DEFORM_1_2
            or special_model_size == EfficientAdModelSize.PDN_M_33_DEFORM_2_4
            or special_model_size == EfficientAdModelSize.PDN_M_33_DEFORM_4_5
        ):
            self.last_upsample = (
                int(img_size[0] / 4) if padding else int(img_size[0] / 4) - 8, # 256, 256
                int(img_size[1] / 4) if padding else int(img_size[1] / 4) - 8, # 256, 256
            )
        elif special_model_size == EfficientAdModelSize.PDN_M_68:
            self.last_upsample = (
                int(img_size[0] / 4) if padding else int(img_size[0] / 8) - 8, # 256,551
                int(img_size[1] / 4) if padding else int(img_size[1] / 8) - 7, # 256,551
            )
        elif special_model_size == EfficientAdModelSize.PDN_M_68_DEFORM:
            self.last_upsample = (
                int(img_size[0] / 4) if padding else int(img_size[0] / 8) - 8, # 416, 896
                int(img_size[1] / 4) if padding else int(img_size[1] / 8) - 8, # 416, 896
            )

        elif special_model_size == EfficientAdModelSize.PDN_M_128:
            self.last_upsample = (
                int(img_size[0] / 4) if padding else int(img_size[0] / 16) - 7, # 256,551
                int(img_size[1] / 4) if padding else int(img_size[1] / 16) - 7, # 256,551
            )
        elif special_model_size == EfficientAdModelSize.PDN_M_128_DEFORM:
            self.last_upsample = (
                int(img_size[0] / 4) if padding else int(img_size[0] / 16) - 7, # 416, 896
                int(img_size[1] / 4) if padding else int(img_size[1] / 16) - 7, # 416, 896
            )
        elif special_model_size == EfficientAdModelSize.PDN_M_256:
            self.last_upsample = (
                int(img_size[0] / 4) if padding else int(img_size[0] / 32) - 7,
                int(img_size[1] / 4) if padding else int(img_size[1] / 32) - 7,
            )
        elif special_model_size == EfficientAdModelSize.PDN_M_256_DEFORM:
            self.last_upsample = (
                int(img_size[0] / 4) if padding else int(img_size[0] / 32) - 7,
                int(img_size[1] / 4) if padding else int(img_size[1] / 32) - 7,
            )

        self.deconv1 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.2)
        self.dropout6 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = F.interpolate(x, size=(int(self.img_size[0] / 64) - 1, int(self.img_size[1] / 64) - 1), mode="bilinear")
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        x = F.interpolate(x, size=(int(self.img_size[0] / 32), int(self.img_size[1] / 32)), mode="bilinear")
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        x = F.interpolate(x, size=(int(self.img_size[0] / 16) - 1, int(self.img_size[1] / 16) - 1), mode="bilinear")
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        x = F.interpolate(x, size=(int(self.img_size[0] / 8), int(self.img_size[1] / 8)), mode="bilinear")
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        x = F.interpolate(x, size=(int(self.img_size[0] / 4) - 1, int(self.img_size[1] / 4) - 1), mode="bilinear")
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        x = F.interpolate(x, size=(int(self.img_size[0] / 2) - 1, int(self.img_size[1] / 2) - 1), mode="bilinear")
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        x = F.interpolate(x, size=self.last_upsample, mode="bilinear")
        x = F.relu(self.deconv7(x))
        x = self.deconv8(x)
        return x


class AutoEncoder(nn.Module):
    """EfficientAd Autoencoder.

    Args:
       out_channels (int): number of convolution output channels
       img_size (tuple): size of input images
    """

    def __init__(self, out_channels, padding, img_size, *args, **kwargs) -> None:

        # exclude special_model_size from kwargs
        special_model_size = kwargs.get("special_model_size", None)
        kwargs = {k: v for k, v in kwargs.items() if k != "special_model_size"}

        super().__init__(*args, **kwargs)
        self.encoder = Encoder()
        self.decoder = Decoder(out_channels, padding, img_size, special_model_size)

    def forward(self, x):
        x = imagenet_norm_batch(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class EfficientAdModel(nn.Module):
    """EfficientAd model.

    Args:
        teacher_out_channels (int): number of convolution output channels of the pre-trained teacher model
        pretrained_models_dir (str): path to the pretrained model weights
        input_size (tuple): size of input images
        model_size (str): size of student and teacher model
        padding (bool): use padding in convoluional layers
        pad_maps (bool): relevant if padding is set to False. In this case, pad_maps = True pads the
            output anomaly maps so that their size matches the size in the padding = True case.
        device (str): which device the model should be loaded on
    """

    def __init__(
        self,
        teacher_out_channels: int,
        input_size: tuple[int, int],
        model_size: EfficientAdModelSize = EfficientAdModelSize.S,
        padding=False,
        pad_maps=True,
        special_model_size = None,
    ) -> None:
        super().__init__()

        self.pad_maps = pad_maps
        self.teacher: PDN_M | PDN_S
        self.student: PDN_M | PDN_S

        if special_model_size == EfficientAdModelSize.PDN_M_16:
            self.teacher = PDN_M_16(out_channels=teacher_out_channels, padding=False).eval()
            self.student = PDN_M_16(out_channels=teacher_out_channels * 2, padding=False)

        elif special_model_size == EfficientAdModelSize.PDN_M_33:
            self.teacher = PDN_M_33(out_channels=teacher_out_channels, padding=False).eval()
            self.student = PDN_M_33(out_channels=teacher_out_channels * 2, padding=False)

        elif special_model_size == EfficientAdModelSize.PDN_M_33_DEFORM:
            self.teacher = PDN_M_33_Deform(out_channels=teacher_out_channels, padding=False).eval()
            self.student = PDN_M_33_Deform(out_channels=teacher_out_channels * 2, padding=False)

        elif special_model_size == EfficientAdModelSize.PDN_M_33_DEFORM_1_2:
            self.teacher = PDN_M_33_Deform_1_2(out_channels=teacher_out_channels, padding=False).eval()
            self.student = PDN_M_33_Deform_1_2(out_channels=teacher_out_channels * 2, padding=False)

        elif special_model_size == EfficientAdModelSize.PDN_M_33_DEFORM_2_4:
            self.teacher = PDN_M_33_Deform_2_4(out_channels=teacher_out_channels, padding=False).eval()
            self.student = PDN_M_33_Deform_2_4(out_channels=teacher_out_channels * 2, padding=False)

        elif special_model_size == EfficientAdModelSize.PDN_M_33_DEFORM_4_5:
            self.teacher = PDN_M_33_Deform_4_5(out_channels=teacher_out_channels, padding=False).eval()
            self.student = PDN_M_33_Deform_4_5(out_channels=teacher_out_channels * 2, padding=False)

        elif special_model_size == EfficientAdModelSize.PDN_M_68:
            self.teacher = PDN_M_68(out_channels=teacher_out_channels, padding=False).eval()
            self.student = PDN_M_68(out_channels=teacher_out_channels * 2, padding=False)

        elif special_model_size == EfficientAdModelSize.PDN_M_68_DEFORM:
            self.teacher = PDN_M_68_Deform(out_channels=teacher_out_channels, padding=False).eval()
            self.student = PDN_M_68_Deform(out_channels=teacher_out_channels * 2, padding=False)

        elif special_model_size == EfficientAdModelSize.PDN_M_128:
            self.teacher = PDN_M_128(out_channels=teacher_out_channels, padding=False).eval()
            self.student = PDN_M_128(out_channels=teacher_out_channels * 2, padding=False)

        elif special_model_size == EfficientAdModelSize.PDN_M_128_DEFORM:
            self.teacher = PDN_M_128_Deform(out_channels=teacher_out_channels, padding=False).eval()
            self.student = PDN_M_128_Deform(out_channels=teacher_out_channels * 2, padding=False)

        elif special_model_size == EfficientAdModelSize.PDN_M_256:
            self.teacher = PDN_M_256(out_channels=teacher_out_channels, padding=False).eval()
            self.student = PDN_M_256(out_channels=teacher_out_channels * 2, padding=False)

        elif special_model_size == EfficientAdModelSize.PDN_M_256_DEFORM:
            self.teacher = PDN_M_256_Deform(out_channels=teacher_out_channels, padding=False).eval()
            self.student = PDN_M_256_Deform(out_channels=teacher_out_channels * 2, padding=False)

        elif model_size == EfficientAdModelSize.M:
            self.teacher = PDN_M(out_channels=teacher_out_channels, padding=padding).eval()
            self.student = PDN_M(out_channels=teacher_out_channels * 2, padding=padding)

        elif model_size == EfficientAdModelSize.S:
            self.teacher = PDN_S(out_channels=teacher_out_channels, padding=padding).eval()
            self.student = PDN_S(out_channels=teacher_out_channels * 2, padding=padding)

        else:
            raise ValueError(f"Unknown model size {model_size}")

        self.ae: AutoEncoder = AutoEncoder(out_channels=teacher_out_channels, padding=padding, img_size=input_size, special_model_size=special_model_size)
        self.teacher_out_channels: int = teacher_out_channels
        self.input_size: tuple[int, int] = input_size

        self.mean_std: nn.ParameterDict = nn.ParameterDict(
            {
                "mean": torch.zeros((1, self.teacher_out_channels, 1, 1)),
                "std": torch.zeros((1, self.teacher_out_channels, 1, 1)),
            }
        )

        self.quantiles: nn.ParameterDict = nn.ParameterDict(
            {
                "qa_st": torch.tensor(0.0),
                "qb_st": torch.tensor(0.0),
                "qa_ae": torch.tensor(0.0),
                "qb_ae": torch.tensor(0.0),
            }
        )

    def is_set(self, p_dic: nn.ParameterDict) -> bool:
        for _, value in p_dic.items():
            if value.sum() != 0:
                return True
        return False

    def choose_random_aug_image(self, image: Tensor) -> Tensor:
        transform_functions = [
            transforms.functional.adjust_brightness,
            transforms.functional.adjust_contrast,
            transforms.functional.adjust_saturation,
        ]
        # Sample an augmentation coefficient λ from the uniform distribution U(0.8, 1.2)
        coefficient = random.uniform(0.8, 1.2)  # nosec: B311
        transform_function = random.choice(transform_functions)  # nosec: B311
        return transform_function(image, coefficient)

    def forward(self, batch: Tensor, batch_imagenet: Tensor = None) -> Tensor | dict:
        """Prediction by EfficientAd models.

        Args:
            batch (Tensor): Input images.

        Returns:
            Tensor: Predictions
        """
        with torch.no_grad():
            teacher_output = self.teacher(batch)
            if self.is_set(self.mean_std):
                teacher_output = (teacher_output - self.mean_std["mean"]) / self.mean_std["std"]

        student_output = self.student(batch)

        distance_st = torch.pow(teacher_output - student_output[:, : self.teacher_out_channels, :, :], 2)

        if self.training:
            # Student loss
            distance_st = reduce_tensor_elems(distance_st)
            d_hard = torch.quantile(distance_st, 0.999)
            loss_hard = torch.mean(distance_st[distance_st >= d_hard])
            student_output_penalty = self.student(batch_imagenet)[:, : self.teacher_out_channels, :, :]
            loss_penalty = torch.mean(student_output_penalty**2)
            loss_st = loss_hard + loss_penalty

            # Autoencoder and Student AE Loss
            aug_img = self.choose_random_aug_image(batch)
            ae_output_aug = self.ae(aug_img)

            with torch.no_grad():
                teacher_output_aug = self.teacher(aug_img)
                if self.is_set(self.mean_std):
                    teacher_output_aug = (teacher_output_aug - self.mean_std["mean"]) / self.mean_std["std"]

            student_output_ae_aug = self.student(aug_img)[:, self.teacher_out_channels :, :, :]

            distance_ae = torch.pow(teacher_output_aug - ae_output_aug, 2)
            distance_stae = torch.pow(ae_output_aug - student_output_ae_aug, 2)

            loss_ae = torch.mean(distance_ae)
            loss_stae = torch.mean(distance_stae)
            return (loss_st, loss_ae, loss_stae)

        else:
            with torch.no_grad():
                ae_output = self.ae(batch)

            map_st = torch.mean(distance_st, dim=1, keepdim=True)
            map_stae = torch.mean(
                (ae_output - student_output[:, self.teacher_out_channels :]) ** 2, dim=1, keepdim=True
            )

            if self.pad_maps:
                map_st = F.pad(map_st, (4, 4, 4, 4))
                map_stae = F.pad(map_stae, (4, 4, 4, 4))
            map_st = F.interpolate(map_st, size=(self.input_size[0], self.input_size[1]), mode="bilinear")
            map_stae = F.interpolate(map_stae, size=(self.input_size[0], self.input_size[1]), mode="bilinear")

            if self.is_set(self.quantiles):
                map_st = 0.1 * (map_st - self.quantiles["qa_st"]) / (self.quantiles["qb_st"] - self.quantiles["qa_st"])
                map_stae = (
                    0.1 * (map_stae - self.quantiles["qa_ae"]) / (self.quantiles["qb_ae"] - self.quantiles["qa_ae"])
                )

            map_combined = 0.5 * map_st + 0.5 * map_stae
            return {"anomaly_map": map_combined, "map_st": map_st, "map_ae": map_stae}


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        # op = (n - (k * d - 1) + 2p / s)
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation)
        return x