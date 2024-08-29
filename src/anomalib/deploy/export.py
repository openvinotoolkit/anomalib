"""Utilities for optimization and OpenVINO conversion."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from enum import Enum

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
