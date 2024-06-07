"""Utilities for optimization and OpenVINO conversion."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from enum import Enum

import torch
from torch import nn
from torchvision.transforms.v2 import CenterCrop, Compose, Resize, Transform

from anomalib.data.transforms import ExportableCenterCrop

logger = logging.getLogger("anomalib")


class ExportType(str, Enum):
    """Model export type.

    Examples:
        >>> from anomalib.deploy import ExportType
        >>> ExportType.ONNX
        'onnx'
        >>> ExportType.OPENVINO
        'openvino'
        >>> ExportType.TORCH
        'torch'
    """

    ONNX = "onnx"
    OPENVINO = "openvino"
    TORCH = "torch"


class CompressionType(str, Enum):
    """Model compression type when exporting to OpenVINO.

    Attributes:
        FP16 (str): Weight compression (FP16). All weights are converted to FP16.
        INT8 (str): Weight compression (INT8). All weights are quantized to INT8,
            but are dequantized to floating point before inference.
        INT8_PTQ (str): Full integer post-training quantization (INT8).
            All weights and operations are quantized to INT8. Inference is done
            in INT8 precision.
        INT8_ACQ (str): Accuracy-control quantization (INT8). Weights and
            operations are quantized to INT8, except those that would degrade
            quality of the model more than is acceptable. Inference is done in
            a mixed precision.

    Examples:
        >>> from anomalib.deploy import CompressionType
        >>> CompressionType.INT8_PTQ
        'int8_ptq'
    """

    FP16 = "fp16"
    INT8 = "int8"
    INT8_PTQ = "int8_ptq"
    INT8_ACQ = "int8_acq"


class InferenceModel(nn.Module):
    """Inference model for export.

    The InferenceModel is used to wrap the model and transform for exporting to torch and ONNX/OpenVINO.

    Args:
        model (nn.Module): Model to export.
        transform (Transform): Input transform for the model.
        disable_antialias (bool, optional): Disable antialiasing in the Resize transforms of the given transform. This
            is needed for ONNX/OpenVINO export, as antialiasing is not supported in the ONNX opset.
    """

    def __init__(self, model: nn.Module, transform: Transform, disable_antialias: bool = False) -> None:
        super().__init__()
        self.model = model
        self.transform = transform
        self.convert_center_crop()
        if disable_antialias:
            self.disable_antialias()

    def forward(self, batch: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Transform the input batch and pass it through the model."""
        batch = self.transform(batch)
        return self.model(batch)

    def disable_antialias(self) -> None:
        """Disable antialiasing in the Resize transforms of the given transform.

        This is needed for ONNX/OpenVINO export, as antialiasing is not supported in the ONNX opset.
        """
        if isinstance(self.transform, Resize):
            self.transform.antialias = False
        if isinstance(self.transform, Compose):
            for transform in self.transform.transforms:
                if isinstance(transform, Resize):
                    transform.antialias = False

    def convert_center_crop(self) -> None:
        """Convert CenterCrop to ExportableCenterCrop for ONNX export.

        The original CenterCrop transform is not supported in ONNX export. This method replaces the CenterCrop to
        ExportableCenterCrop, which is supported in ONNX export. For more details, see the implementation of
        ExportableCenterCrop.
        """
        if isinstance(self.transform, CenterCrop):
            self.transform = ExportableCenterCrop(size=self.transform.size)
        elif isinstance(self.transform, Compose):
            transforms = self.transform.transforms
            for index in range(len(transforms)):
                if isinstance(transforms[index], CenterCrop):
                    transforms[index] = ExportableCenterCrop(size=transforms[index].size)
