"""Numpy-based video dataclasses for Anomalib.

This module provides numpy-based implementations of video-specific dataclasses used in
Anomalib. These classes are designed to work with video data represented as numpy
arrays for anomaly detection tasks.

The module contains two main classes:
    - :class:`NumpyVideoItem`: For single video data items
    - :class:`NumpyVideoBatch`: For batched video data items

Example:
    Create and use a numpy video item:

    >>> from anomalib.data.dataclasses.numpy import NumpyVideoItem
    >>> import numpy as np
    >>> item = NumpyVideoItem(
    ...     data=np.random.rand(16, 224, 224, 3),  # (T, H, W, C)
    ...     frames=np.random.rand(16, 224, 224, 3),
    ...     label=0,
    ...     video_path="path/to/video.mp4"
    ... )
    >>> item.frames.shape
    (16, 224, 224, 3)
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np

from anomalib.data.dataclasses.generic import BatchIterateMixin, _VideoInputFields
from anomalib.data.dataclasses.numpy.base import NumpyBatch, NumpyItem
from anomalib.data.validators.numpy.video import NumpyVideoBatchValidator, NumpyVideoValidator


@dataclass
class NumpyVideoItem(
    NumpyVideoValidator,
    _VideoInputFields[np.ndarray, np.ndarray, np.ndarray, str],
    NumpyItem,
):
    """Dataclass for a single video item in Anomalib datasets using numpy arrays.

    This class combines :class:`_VideoInputFields` and :class:`NumpyItem` for
    video-based anomaly detection. It includes video-specific fields and validation
    methods to ensure proper formatting for Anomalib's video-based models.

    The class uses the following type parameters:
        - Video: :class:`numpy.ndarray` with shape ``(T, H, W, C)``
        - Label: :class:`numpy.ndarray`
        - Mask: :class:`numpy.ndarray` with shape ``(T, H, W)``
        - Path: :class:`str`

    Where ``T`` represents the temporal dimension (number of frames).
    """


@dataclass
class NumpyVideoBatch(
    BatchIterateMixin[NumpyVideoItem],
    NumpyVideoBatchValidator,
    _VideoInputFields[np.ndarray, np.ndarray, np.ndarray, list[str]],
    NumpyBatch,
):
    """Dataclass for a batch of video items in Anomalib datasets using numpy arrays.

    This class combines :class:`BatchIterateMixin`, :class:`_VideoInputFields`, and
    :class:`NumpyBatch` for batches of video data. It supports batch operations
    and iteration over individual :class:`NumpyVideoItem` instances.

    The class uses the following type parameters:
        - Video: :class:`numpy.ndarray` with shape ``(B, T, H, W, C)``
        - Label: :class:`numpy.ndarray` with shape ``(B,)``
        - Mask: :class:`numpy.ndarray` with shape ``(B, T, H, W)``
        - Path: :class:`list` of :class:`str`

    Where ``B`` represents the batch dimension and ``T`` the temporal dimension.
    """

    item_class = NumpyVideoItem
