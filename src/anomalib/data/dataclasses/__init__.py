# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Anomalib dataclasses.

This module provides a collection of dataclasses used throughout the Anomalib
library for representing and managing various types of data related to anomaly
detection tasks.

The dataclasses are organized into two main categories:
1. Numpy-based dataclasses for handling numpy array data
2. Torch-based dataclasses for handling PyTorch tensor data

Key Components
-------------

Numpy Dataclasses
~~~~~~~~~~~~~~~~

- :class:`NumpyImageItem`: Single image item as numpy arrays
    - Data shape: ``(H, W, C)`` or ``(H, W)`` for grayscale
    - Labels: Binary classification (0: normal, 1: anomalous)
    - Masks: Binary segmentation masks ``(H, W)``

- :class:`NumpyImageBatch`: Batch of image data as numpy arrays
    - Data shape: ``(N, H, W, C)`` or ``(N, H, W)`` for grayscale
    - Labels: ``(N,)`` binary labels
    - Masks: ``(N, H, W)`` binary masks

- :class:`NumpyVideoItem`: Single video item as numpy arrays
    - Data shape: ``(T, H, W, C)`` or ``(T, H, W)`` for grayscale
    - Labels: Binary classification per video
    - Masks: ``(T, H, W)`` temporal segmentation masks

- :class:`NumpyVideoBatch`: Batch of video data as numpy arrays
    - Data shape: ``(N, T, H, W, C)`` or ``(N, T, H, W)`` for grayscale
    - Labels: ``(N,)`` binary labels
    - Masks: ``(N, T, H, W)`` batch of temporal masks

Torch Dataclasses
~~~~~~~~~~~~~~~~

- :class:`Batch`: Base class for torch-based batch data
- :class:`DatasetItem`: Base class for torch-based dataset items
- :class:`DepthItem`: Single depth data item
    - RGB image: ``(3, H, W)``
    - Depth map: ``(H, W)``
- :class:`DepthBatch`: Batch of depth data
    - RGB images: ``(N, 3, H, W)``
    - Depth maps: ``(N, H, W)``
- :class:`ImageItem`: Single image as torch tensors
    - Data shape: ``(C, H, W)``
- :class:`ImageBatch`: Batch of images as torch tensors
    - Data shape: ``(N, C, H, W)``
- :class:`VideoItem`: Single video as torch tensors
    - Data shape: ``(T, C, H, W)``
- :class:`VideoBatch`: Batch of videos as torch tensors
    - Data shape: ``(N, T, C, H, W)``
- :class:`InferenceBatch`: Specialized batch for inference results
    - Predictions: Scores, labels, anomaly maps and masks

These dataclasses provide a structured way to handle various types of data
in anomaly detection tasks, ensuring type consistency and easy data manipulation
across different components of the Anomalib library.

Example:
-------
>>> from anomalib.data.dataclasses import ImageItem
>>> import torch
>>> item = ImageItem(
...     image=torch.rand(3, 224, 224),
...     gt_label=torch.tensor(0),
...     image_path="path/to/image.jpg"
... )
>>> item.image.shape
torch.Size([3, 224, 224])
"""

from .numpy import NumpyImageBatch, NumpyImageItem, NumpyVideoBatch, NumpyVideoItem
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
    # Numpy
    "NumpyImageItem",
    "NumpyImageBatch",
    "NumpyVideoItem",
    "NumpyVideoBatch",
    # Torch
    "DatasetItem",
    "Batch",
    "InferenceBatch",
    "ImageItem",
    "ImageBatch",
    "VideoItem",
    "VideoBatch",
    "DepthItem",
    "DepthBatch",
]
