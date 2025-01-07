"""Utilities for optimization and OpenVINO conversion.

This module provides functionality for exporting and optimizing anomaly detection
models to different formats like ONNX, OpenVINO IR and PyTorch.

Example:
    Export a model to ONNX format:

    >>> from anomalib.deploy import ExportType
    >>> export_type = ExportType.ONNX
    >>> export_type
    'onnx'

    Export with OpenVINO compression:

    >>> from anomalib.deploy import CompressionType
    >>> compression = CompressionType.INT8_PTQ
    >>> compression
    'int8_ptq'
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from enum import Enum

logger = logging.getLogger("anomalib")


class ExportType(str, Enum):
    """Model export type.

    Supported export formats for anomaly detection models.

    Attributes:
        ONNX: Export model to ONNX format
        OPENVINO: Export model to OpenVINO IR format
        TORCH: Export model to PyTorch format

    Example:
        >>> from anomalib.deploy import ExportType
        >>> export_type = ExportType.ONNX
        >>> export_type
        'onnx'
    """

    ONNX = "onnx"
    OPENVINO = "openvino"
    TORCH = "torch"


class CompressionType(str, Enum):
    """Model compression type when exporting to OpenVINO.

    Attributes:
        FP16: Weight compression to FP16 precision. All weights are converted
            to FP16.
        INT8: Weight compression to INT8 precision. All weights are quantized
            to INT8, but are dequantized to floating point before inference.
        INT8_PTQ: Full integer post-training quantization to INT8 precision.
            All weights and operations are quantized to INT8. Inference is
            performed in INT8 precision.
        INT8_ACQ: Accuracy-control quantization to INT8 precision. Weights and
            operations are quantized to INT8, except those that would degrade
            model quality beyond an acceptable threshold. Inference uses mixed
            precision.

    Example:
        >>> from anomalib.deploy import CompressionType
        >>> compression = CompressionType.INT8_PTQ
        >>> compression
        'int8_ptq'
    """

    FP16 = "fp16"
    INT8 = "int8"
    INT8_PTQ = "int8_ptq"
    INT8_ACQ = "int8_acq"
