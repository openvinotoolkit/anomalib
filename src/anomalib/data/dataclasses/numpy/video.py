"""Numpy-based video dataclasses for Anomalib."""

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

    This class combines _VideoInputFields and NumpyItem for video-based anomaly detection.
    It includes video-specific fields and validation methods to ensure proper formatting
    for Anomalib's video-based models.
    """


@dataclass
class NumpyVideoBatch(
    BatchIterateMixin[NumpyVideoItem],
    NumpyVideoBatchValidator,
    _VideoInputFields[np.ndarray, np.ndarray, np.ndarray, list[str]],
    NumpyBatch,
):
    """Dataclass for a batch of video items in Anomalib datasets using numpy arrays.

    This class combines BatchIterateMixin, _VideoInputFields, and NumpyBatch for batches
    of video data. It supports batch operations and iteration over individual NumpyVideoItems.
    It ensures proper formatting for Anomalib's video-based models.
    """

    item_class = NumpyVideoItem
