"""Numpy-based image dataclasses for Anomalib.

This module provides numpy-based implementations of image-specific dataclasses used in
Anomalib. These classes are designed to work with image data represented as numpy arrays
for anomaly detection tasks.

The module contains two main classes:
    - :class:`NumpyImageItem`: For single image data items
    - :class:`NumpyImageBatch`: For batched image data items

Example:
    Create and use a numpy image item::

        >>> from anomalib.data.dataclasses.numpy import NumpyImageItem
        >>> import numpy as np
        >>> item = NumpyImageItem(
        ...     data=np.random.rand(224, 224, 3),
        ...     label=0,
        ...     image_path="path/to/image.jpg"
        ... )
        >>> item.data.shape
        (224, 224, 3)
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from anomalib.data.dataclasses.generic import BatchIterateMixin, _ImageInputFields
from anomalib.data.dataclasses.numpy.base import NumpyBatch, NumpyItem
from anomalib.data.validators.numpy.image import NumpyImageBatchValidator, NumpyImageValidator


@dataclass
class NumpyImageItem(
    NumpyImageValidator,
    _ImageInputFields[str],
    NumpyItem,
):
    """Dataclass for a single image item in Anomalib datasets using numpy arrays.

    This class combines :class:`_ImageInputFields` and :class:`NumpyItem` for
    image-based anomaly detection. It includes image-specific fields and validation
    methods to ensure proper formatting for Anomalib's image-based models.

    The class uses the following type parameters:
        - Image: :class:`numpy.ndarray` with shape ``(H, W, C)``
        - Label: :class:`numpy.ndarray`
        - Mask: :class:`numpy.ndarray` with shape ``(H, W)``
        - Path: :class:`str`

    Example:
        >>> import numpy as np
        >>> from anomalib.data.dataclasses.numpy import NumpyImageItem
        >>> item = NumpyImageItem(
        ...     data=np.random.rand(224, 224, 3),
        ...     label=0,
        ...     image_path="path/to/image.jpg"
        ... )
        >>> item.data.shape
        (224, 224, 3)
    """


@dataclass
class NumpyImageBatch(
    BatchIterateMixin[NumpyImageItem],
    NumpyImageBatchValidator,
    _ImageInputFields[list[str]],
    NumpyBatch,
):
    """Dataclass for a batch of image items in Anomalib datasets using numpy arrays.

    This class combines :class:`BatchIterateMixin`, :class:`_ImageInputFields`, and
    :class:`NumpyBatch` for batches of image data. It supports batch operations and
    iteration over individual :class:`NumpyImageItem` instances.

    The class uses the following type parameters:
        - Image: :class:`numpy.ndarray` with shape ``(B, H, W, C)``
        - Label: :class:`numpy.ndarray` with shape ``(B,)``
        - Mask: :class:`numpy.ndarray` with shape ``(B, H, W)``
        - Path: :class:`list` of :class:`str`

    Where ``B`` represents the batch dimension that is prepended to all tensor-like
    fields.

    Example:
        >>> import numpy as np
        >>> from anomalib.data.dataclasses.numpy import NumpyImageBatch
        >>> batch = NumpyImageBatch(
        ...     data=np.random.rand(32, 224, 224, 3),
        ...     label=np.zeros(32),
        ...     image_path=[f"path/to/image_{i}.jpg" for i in range(32)]
        ... )
        >>> batch.data.shape
        (32, 224, 224, 3)
    """

    item_class = NumpyImageItem
