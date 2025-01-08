"""Functions for model inference and deployment.

This module provides functionality for deploying trained anomaly detection models
and performing inference. It includes:

- Model export utilities for converting models to different formats
- Inference classes for making predictions:
    - :class:`Inferencer`: Base inferencer interface
    - :class:`TorchInferencer`: For PyTorch models
    - :class:`OpenVINOInferencer`: For OpenVINO IR models

Example:
    >>> from anomalib.deploy import TorchInferencer
    >>> model = TorchInferencer(path="path/to/model.pt")
    >>> predictions = model.predict(image="path/to/image.jpg")

    The prediction contains anomaly maps and scores:

    >>> predictions.anomaly_map  # doctest: +SKIP
    tensor([[0.1, 0.2, ...]])
    >>> predictions.pred_score  # doctest: +SKIP
    tensor(0.86)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .export import CompressionType, ExportType
from .inferencers import Inferencer, OpenVINOInferencer, TorchInferencer

__all__ = ["Inferencer", "OpenVINOInferencer", "TorchInferencer", "ExportType", "CompressionType"]
