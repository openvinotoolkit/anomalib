"""Torch model for student, teacher and autoencoder model in EfficientAD"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision import transforms

logger = logging.getLogger(__name__)


def imagenet_norm_batch(x):
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].cuda()
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].cuda()
    x_norm = (x - mean) / (std + 1e-11)
    return x_norm


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

    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0 * pad_mult)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x):
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
    """

    def __init__(self, out_channels, padding, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_upsample = 64 if padding else 56
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
        x = F.interpolate(x, size=3, mode="bilinear")
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        x = F.interpolate(x, size=8, mode="bilinear")
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        x = F.interpolate(x, size=15, mode="bilinear")
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        x = F.interpolate(x, size=32, mode="bilinear")
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        x = F.interpolate(x, size=63, mode="bilinear")
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        x = F.interpolate(x, size=127, mode="bilinear")
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        x = F.interpolate(x, size=self.last_upsample, mode="bilinear")
        x = F.relu(self.deconv7(x))
        x = self.deconv8(x)
        return x


class AutoEncoder(nn.Module):
    """EfficientAD Autoencoder.

    Args:
       out_channels (int): number of convolution output channels
    """

    def __init__(self, out_channels, padding, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = Encoder()
        self.decoder = Decoder(out_channels, padding)

    def forward(self, x):
        x = imagenet_norm_batch(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class EfficientADModel(nn.Module):
    """EfficientAD model.

    Args:
        teacher_path (Path): path of pre-trained teacher model.
        teacher_out_channels (int): number of convolution output channels of the pre-trained teacher model
        model_size (str): size of teacher and student model
    """

    def __init__(
        self,
        teacher_path: Path,
        teacher_out_channels: int,
        input_size: list,
        model_size="M",
        padding=False,
    ) -> None:
        super().__init__()

        self.teacher: Union[PDN_M, PDN_S]
        self.student: Union[PDN_M, PDN_S]

        if model_size == "M":
            self.teacher = PDN_M(out_channels=teacher_out_channels, padding=padding).eval()
            self.student = PDN_M(out_channels=teacher_out_channels * 2, padding=padding)

        elif model_size == "S":
            self.teacher = PDN_S(out_channels=teacher_out_channels, padding=padding).eval()
            self.student = PDN_S(out_channels=teacher_out_channels * 2, padding=padding)

        else:
            raise ValueError(f"Unknown model size {model_size}")

        logger.info(f"Load pretrained Teacher model from {teacher_path}")
        self.teacher.load_state_dict(torch.load(teacher_path))

        self.ae: AutoEncoder = AutoEncoder(out_channels=teacher_out_channels, padding=padding)
        self.teacher_out_channels: int = teacher_out_channels
        self.input_size: list[int] = input_size

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
        aug_index = random.choice([1, 2, 3])
        # Sample an augmentation coefficient λ from the uniform distribution U(0.8, 1.2)
        coefficient = random.uniform(0.8, 1.2)
        if aug_index == 1:
            img_aug = transforms.functional.adjust_brightness(image, coefficient)
        elif aug_index == 2:
            img_aug = transforms.functional.adjust_contrast(image, coefficient)
        elif aug_index == 3:
            img_aug = transforms.functional.adjust_saturation(image, coefficient)
        return img_aug

    def forward(self, batch: Tensor, batch_imagenet: Tensor = None) -> Union[Tensor, dict]:
        """Prediction by EfficientAD models.

        Args:
            batch (Tensor): Input images.

        Returns:
            Tensor: Predictions
        """
        with torch.no_grad():
            teacher_output = self.teacher(batch)
            if self.is_set(self.mean_std):
                teacher_output = (teacher_output - self.mean_std["mean"]) / self.mean_std["std"]
            _, c, h, w = teacher_output.shape

        student_output = self.student(batch)
        ae_output = self.ae(batch)
        # 3: Split the student output into Y ST ∈ R 384×64×64 and Y STAE ∈ R 384×64×64 as above
        student_output = student_output[:, : self.teacher_out_channels, :, :]
        student_output_ae = student_output[:, -self.teacher_out_channels :, :, :]

        distance_st = torch.pow(teacher_output - student_output, 2)

        if self.training:
            # Student loss
            d_hard = torch.quantile(distance_st, 0.999)
            loss_hard = torch.mean(distance_st[distance_st >= d_hard])
            student_imagenet_output = self.student(batch_imagenet)[:, : self.teacher_out_channels, :, :]
            loss_st = loss_hard + (1 / (c * h * w)) * torch.sum(torch.pow(student_imagenet_output, 2))

            # Autoencoder and Student AE Loss
            aug_img = self.choose_random_aug_image(batch)
            ae_output_aug = self.ae(aug_img)
            student_output_aug = self.student(aug_img)
            student_output_ae_aug = student_output_aug[:, -self.teacher_out_channels :, :, :]

            with torch.no_grad():
                teacher_output_aug = self.teacher(aug_img)
                if self.is_set(self.mean_std):
                    teacher_output_aug = (teacher_output_aug - self.mean_std["mean"]) / self.mean_std["std"]

            distance_ae = torch.pow(teacher_output_aug - ae_output_aug, 2)
            distance_stae = torch.pow(ae_output_aug - student_output_ae_aug, 2)

            loss_ae = torch.mean(distance_ae) *10
            loss_stae = torch.mean(distance_stae) *100

            return (loss_st, loss_ae, loss_stae)

        else:
            distance_stae = torch.pow(ae_output - student_output_ae, 2)

            map_st = torch.mean(distance_st, dim=1, keepdim=True)
            map_stae = torch.mean(distance_stae, dim=1, keepdim=True)

            map_st = F.interpolate(map_st, size=(self.input_size[0], self.input_size[1]), mode="bilinear")
            map_stae = F.interpolate(map_stae, size=(self.input_size[0], self.input_size[1]), mode="bilinear")

            if self.is_set(self.quantiles):
                map_st = 0.1 * (map_st - self.quantiles["qa_st"]) / (self.quantiles["qb_st"] - self.quantiles["qa_st"])
                map_stae = (
                    0.1 * (map_stae - self.quantiles["qa_ae"]) / (self.quantiles["qb_ae"] - self.quantiles["qa_ae"])
                )

            map_combined = 0.5 * map_st + 0.5 * map_stae

            return {"anomaly_map_combined": map_combined, "map_st": map_st, "map_ae": map_stae}
