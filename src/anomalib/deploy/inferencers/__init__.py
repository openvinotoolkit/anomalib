"""Inferencers for performing inference with anomaly detection models.

This module provides inferencer classes for running inference with trained models
using different backends:

- :class:`Inferencer`: Base class defining the inferencer interface
- :class:`TorchInferencer`: For inference with PyTorch models
- :class:`OpenVINOInferencer`: For optimized inference with OpenVINO

Example:
    >>> from anomalib.deploy import TorchInferencer
    >>> model = TorchInferencer(path="path/to/model.pt")
    >>> predictions = model.predict(image="path/to/image.jpg")
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .base_inferencer import Inferencer
from .openvino_inferencer import OpenVINOInferencer
from .torch_inferencer import TorchInferencer

__all__ = ["Inferencer", "OpenVINOInferencer", "TorchInferencer"]
