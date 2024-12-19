"""Numpy-based depth dataclasses for Anomalib.

This module provides numpy-based implementations of depth-specific dataclasses used in
Anomalib. These classes are designed to work with depth data represented as numpy arrays
for anomaly detection tasks.

The module contains two main classes:
    - :class:`NumpyDepthItem`: For single depth data items
    - :class:`NumpyDepthBatch`: For batched depth data items

Example:
    Create and use a numpy depth item:

    >>> from anomalib.data.dataclasses.numpy import NumpyDepthItem
    >>> import numpy as np
    >>> item = NumpyDepthItem(
    ...     data=np.random.rand(224, 224, 1),
    ...     depth=np.random.rand(224, 224),
    ...     label=0,
    ...     depth_path="path/to/depth.png"
    ... )
    >>> item.depth.shape
    (224, 224)
"""

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

    This class combines :class:`_DepthInputFields` and :class:`NumpyItem` for
    depth-based anomaly detection. It includes depth-specific fields and validation
    methods to ensure proper formatting for Anomalib's depth-based models.

    The class uses the following type parameters:
        - Image: :class:`numpy.ndarray` with shape ``(H, W, C)``
        - Depth: :class:`numpy.ndarray` with shape ``(H, W)``
        - Label: :class:`numpy.ndarray`
        - Path: :class:`str`

    Example:
        >>> import numpy as np
        >>> from anomalib.data.dataclasses.numpy import NumpyDepthItem
        >>> item = NumpyDepthItem(
        ...     data=np.random.rand(224, 224, 3),
        ...     depth=np.random.rand(224, 224),
        ...     label=0,
        ...     depth_path="path/to/depth.png"
        ... )
    """


class NumpyDepthBatch(
    BatchIterateMixin[NumpyDepthItem],
    NumpyDepthBatchValidator,
    _DepthInputFields[np.ndarray, list[str]],
    NumpyBatch,
):
    """Dataclass for a batch of depth items in Anomalib datasets using numpy arrays.

    This class extends :class:`NumpyBatch` for batches of depth-based data. It
    represents multiple depth data points for batch processing in anomaly detection
    tasks.

    The class uses the following type parameters:
        - Image: :class:`numpy.ndarray` with shape ``(B, C, H, W)``
        - Depth: :class:`numpy.ndarray` with shape ``(B, H, W)``
        - Label: :class:`numpy.ndarray` with shape ``(B,)``
        - Path: :class:`list` of :class:`str`

    Where ``B`` represents the batch dimension that is prepended to all
    tensor-like fields.
    """

    item_class = NumpyDepthItem
