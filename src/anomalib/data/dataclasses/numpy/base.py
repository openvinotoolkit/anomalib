"""Numpy-based dataclasses for Anomalib.

This module provides numpy-based implementations of the generic dataclasses
used in Anomalib. These classes are designed to work with numpy arrays for
efficient data handling and processing in anomaly detection tasks.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np

from anomalib.data.generic import _GenericBatch, _GenericItem


@dataclass
class NumpyItem(_GenericItem[np.ndarray, np.ndarray, np.ndarray, str]):
    """Dataclass for a single item in Anomalib datasets using numpy arrays.

    This class extends _GenericItem for numpy-based data representation. It includes
    both input data (e.g., images, labels) and output data (e.g., predictions,
    anomaly maps) as numpy arrays. It is suitable for numpy-based processing
    pipelines in Anomalib.
    """


@dataclass
class NumpyBatch(_GenericBatch[np.ndarray, np.ndarray, np.ndarray, list[str]]):
    """Dataclass for a batch of items in Anomalib datasets using numpy arrays.

    This class extends _GenericBatch for batches of numpy-based data. It represents
    multiple data points for batch processing in anomaly detection tasks. It includes
    an additional dimension for batch size in all tensor-like fields.
    """
