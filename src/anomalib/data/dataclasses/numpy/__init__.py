"""Numpy-based dataclasses for Anomalib.

This module provides numpy-based implementations of the generic dataclasses used in
Anomalib. These classes are designed to work with numpy arrays for efficient data
handling and processing in anomaly detection tasks.

The module includes the following main classes:

- :class:`NumpyItem`: Base class representing a single item in Anomalib datasets
  using numpy arrays. Contains common fields like ``data``, ``label``,
  ``label_index``, ``split``, and ``metadata``.

- :class:`NumpyBatch`: Base class representing a batch of items in Anomalib
  datasets using numpy arrays. Provides batch operations and collation
  functionality.

- :class:`NumpyImageItem`: Specialized class for image data that extends
  :class:`NumpyItem` with image-specific fields like ``image_path``, ``mask``,
  ``mask_path``, ``anomaly_maps``, and ``boxes``.

- :class:`NumpyImageBatch`: Specialized batch class for image data that extends
  :class:`NumpyBatch` with image-specific batch operations and collation.

- :class:`NumpyVideoItem`: Specialized class for video data that extends
  :class:`NumpyItem` with video-specific fields like ``video_path``, ``frames``,
  ``frame_masks``, and ``frame_boxes``.

- :class:`NumpyVideoBatch`: Specialized batch class for video data that extends
  :class:`NumpyBatch` with video-specific batch operations and collation.

Example:
    Create and use a numpy image item:

    >>> from anomalib.data.dataclasses.numpy import NumpyImageItem
    >>> import numpy as np
    >>> item = NumpyImageItem(
    ...     data=np.random.rand(224, 224, 3),
    ...     label=0,
    ...     image_path="path/to/image.jpg"
    ... )
    >>> item.data.shape
    (224, 224, 3)

Note:
    - All classes in this module use numpy arrays internally for efficient data
      handling
    - The batch classes provide automatic collation of items into batches suitable
      for model input
    - The classes are designed to be compatible with Anomalib's data pipeline and
      model interfaces
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .base import NumpyBatch, NumpyItem
from .image import NumpyImageBatch, NumpyImageItem
from .video import NumpyVideoBatch, NumpyVideoItem

__all__ = [
    "NumpyBatch",
    "NumpyItem",
    "NumpyImageBatch",
    "NumpyImageItem",
    "NumpyVideoBatch",
    "NumpyVideoItem",
]
