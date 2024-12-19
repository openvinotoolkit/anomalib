"""Anomalib Numpy data validators.

This module provides validators for numpy array data used in Anomalib. The validators
ensure data consistency and correctness for various data types:

- Image data: Single images and batches
- Video data: Single videos and batches
- Depth data: Single depth maps and batches

The validators check:
    - Array shapes and dimensions
    - Data types
    - Value ranges
    - Label formats
    - Mask properties

Example:
    Validate a numpy image batch::

        >>> from anomalib.data.validators import NumpyImageBatchValidator
        >>> validator = NumpyImageBatchValidator()
        >>> validator(images=images, labels=labels, masks=masks)

Note:
    The validators are used internally by the data modules to ensure data
    consistency before processing.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .depth import NumpyDepthBatchValidator, NumpyDepthValidator
from .image import NumpyImageBatchValidator, NumpyImageValidator
from .video import NumpyVideoBatchValidator, NumpyVideoValidator

__all__ = [
    "NumpyImageBatchValidator",
    "NumpyImageValidator",
    "NumpyVideoBatchValidator",
    "NumpyVideoValidator",
    "NumpyDepthBatchValidator",
    "NumpyDepthValidator",
]
