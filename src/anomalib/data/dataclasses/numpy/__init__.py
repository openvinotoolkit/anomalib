"""Numpy-based dataclasses for Anomalib.

This module provides numpy-based implementations of the generic dataclasses
used in Anomalib. These classes are designed to work with numpy arrays for
efficient data handling and processing in anomaly detection tasks.

The module includes the following main classes:

- NumpyItem: Represents a single item in Anomalib datasets using numpy arrays.
- NumpyBatch: Represents a batch of items in Anomalib datasets using numpy arrays.
- NumpyImageItem: Represents a single image item with additional image-specific fields.
- NumpyImageBatch: Represents a batch of image items with batch operations.
- NumpyVideoItem: Represents a single video item with video-specific fields.
- NumpyVideoBatch: Represents a batch of video items with video-specific operations.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .base import NumpyBatch, NumpyItem
from .image import NumpyImageBatch, NumpyImageItem
from .video import NumpyVideoBatch, NumpyVideoItem

__all__ = ["NumpyBatch", "NumpyItem", "NumpyImageBatch", "NumpyImageItem", "NumpyVideoBatch", "NumpyVideoItem"]
