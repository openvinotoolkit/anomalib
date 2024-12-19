"""Validate PyTorch tensor data.

This module provides validators for data stored as PyTorch tensors. The validators
ensure data consistency and correctness for images, videos, depth maps and their
batches.

The validators check:
    - Tensor shapes and dimensions
    - Data types
    - Value ranges
    - Label formats
    - Mask properties

Example:
    Validate a single image::

        >>> from anomalib.data.validators import ImageValidator
        >>> validator = ImageValidator()
        >>> validator.validate_image(image)

    Validate a batch of images::

        >>> from anomalib.data.validators import ImageBatchValidator
        >>> validator = ImageBatchValidator()
        >>> validator(images=images, labels=labels, masks=masks)

Note:
    The validators are used internally by the data modules to ensure data
    consistency before processing.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .depth import DepthBatchValidator, DepthValidator
from .image import ImageBatchValidator, ImageValidator
from .video import VideoBatchValidator, VideoValidator

__all__ = [
    "DepthBatchValidator",
    "DepthValidator",
    "ImageBatchValidator",
    "ImageValidator",
    "VideoBatchValidator",
    "VideoValidator",
]
