"""Numpy-based depth dataclasses for Anomalib."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import numpy as np

from anomalib.data.dataclasses.generic import BatchIterateMixin, _DepthInputFields
from anomalib.data.dataclasses.numpy.base import NumpyBatch, NumpyItem
from anomalib.data.validators.numpy.depth import NumpyDepthBatchValidator, NumpyDepthValidator


@dataclass
class NumpyDepthItem(
    NumpyDepthValidator,
    _DepthInputFields[np.ndarray, str],
    NumpyItem,
):
    """Dataclass for a single depth item in Anomalib datasets using numpy arrays.

    This class combines _DepthInputFields and NumpyItem for depth-based anomaly detection.
    It includes depth-specific fields and validation methods to ensure proper formatting
    for Anomalib's depth-based models.
    """


class NumpyDepthBatch(
    BatchIterateMixin[NumpyDepthItem],
    NumpyDepthBatchValidator,
    _DepthInputFields[np.ndarray, list[str]],
    NumpyBatch,
):
    """Dataclass for a batch of depth items in Anomalib datasets using numpy arrays."""

    item_class = NumpyDepthItem
