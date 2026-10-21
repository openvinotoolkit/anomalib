# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch-based dataclasses for Anomalib.

This module provides PyTorch-based implementations of the generic dataclasses
used in Anomalib. These classes are designed to work with PyTorch tensors for
efficient data handling and processing in anomaly detection tasks.

These classes extend the generic dataclasses defined in the Anomalib framework,
providing concrete implementations that use PyTorch tensors for tensor-like data.
They include methods for data validation and support operations specific to
image, video, and depth data in the context of anomaly detection.

Note:
    When using these classes, ensure that the input data is in the correct
    format (PyTorch tensors with appropriate shapes) to avoid validation errors.
"""

from .base import Batch, DatasetItem, InferenceBatch, ToNumpyMixin
from .depth import DepthBatch, DepthItem
from .image import ImageBatch, ImageItem
from .video import VideoBatch, VideoItem

__all__ = [
    # Base
    "Batch",
    "DatasetItem",
    "InferenceBatch",
    "ToNumpyMixin",
    # Depth
    "DepthItem",
    "DepthBatch",
    # Image
    "ImageItem",
    "ImageBatch",
    # Video
    "VideoItem",
    "VideoBatch",
]
