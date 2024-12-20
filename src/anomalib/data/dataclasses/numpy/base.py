"""Numpy-based dataclasses for Anomalib.

This module provides numpy-based implementations of the generic dataclasses used in
Anomalib. These classes are designed to work with :class:`numpy.ndarray` objects
for efficient data handling and processing in anomaly detection tasks.

The module contains two main classes:
    - :class:`NumpyItem`: For single data items
    - :class:`NumpyBatch`: For batched data items
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np

from anomalib.data.dataclasses.generic import _GenericBatch, _GenericItem


@dataclass
class NumpyItem(_GenericItem[np.ndarray, np.ndarray, np.ndarray, str]):
    """Dataclass for a single item in Anomalib datasets using numpy arrays.

    This class extends :class:`_GenericItem` for numpy-based data representation.
    It includes both input data (e.g., images, labels) and output data (e.g.,
    predictions, anomaly maps) as numpy arrays.

    The class uses the following type parameters:
        - Image: :class:`numpy.ndarray`
        - Label: :class:`numpy.ndarray`
        - Mask: :class:`numpy.ndarray`
        - Path: :class:`str`

    This implementation is suitable for numpy-based processing pipelines in
    Anomalib where GPU acceleration is not required.
    """


@dataclass
class NumpyBatch(_GenericBatch[np.ndarray, np.ndarray, np.ndarray, list[str]]):
    """Dataclass for a batch of items in Anomalib datasets using numpy arrays.

    This class extends :class:`_GenericBatch` for batches of numpy-based data.
    It represents multiple data points for batch processing in anomaly detection
    tasks.

    The class uses the following type parameters:
        - Image: :class:`numpy.ndarray` with shape ``(B, C, H, W)``
        - Label: :class:`numpy.ndarray` with shape ``(B,)``
        - Mask: :class:`numpy.ndarray` with shape ``(B, H, W)``
        - Path: :class:`list` of :class:`str`

    Where ``B`` represents the batch dimension that is prepended to all
    tensor-like fields.
    """
