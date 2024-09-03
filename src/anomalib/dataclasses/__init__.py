"""Anomalib dataclasses.

This module provides a collection of dataclasses used throughout the Anomalib library
for representing and managing various types of data related to anomaly detection tasks.

The dataclasses are organized into two main categories:
1. Numpy-based dataclasses for handling numpy array data.
2. Torch-based dataclasses for handling PyTorch tensor data.

Key components:

Numpy Dataclasses:
    ``NumpyImageItem``: Represents a single image item as numpy arrays.
    ``NumpyImageBatch``: Represents a batch of image data as numpy arrays.
    ``NumpyVideoItem``: Represents a single video item as numpy arrays.
    ``NumpyVideoBatch``: Represents a batch of video data as numpy arrays.

Torch Dataclasses:
    ``Batch``: Base class for torch-based batch data.
    ``DatasetItem``: Base class for torch-based dataset items.
    ``DepthItem``: Represents a single depth data item.
    ``DepthBatch``: Represents a batch of depth data.
    ``ImageItem``: Represents a single image item as torch tensors.
    ``ImageBatch``: Represents a batch of image data as torch tensors.
    ``VideoItem``: Represents a single video item as torch tensors.
    ``VideoBatch``: Represents a batch of video data as torch tensors.
    ``InferenceBatch``: Specialized batch class for inference results.

These dataclasses provide a structured way to handle various types of data
in anomaly detection tasks, ensuring type consistency and easy data manipulation
across different components of the Anomalib library.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .numpy import (
    NumpyImageBatch,
    NumpyImageItem,
    NumpyVideoBatch,
    NumpyVideoItem,
)
from .torch import (
    Batch,
    DatasetItem,
    DepthBatch,
    DepthItem,
    ImageBatch,
    ImageItem,
    InferenceBatch,
    VideoBatch,
    VideoItem,
)

__all__ = [
    "DatasetItem",
    "Batch",
    "InferenceBatch",
    "ImageItem",
    "ImageBatch",
    "VideoItem",
    "VideoBatch",
    "NumpyImageItem",
    "NumpyImageBatch",
    "NumpyVideoItem",
    "NumpyVideoBatch",
    "DepthItem",
    "DepthBatch",
]
