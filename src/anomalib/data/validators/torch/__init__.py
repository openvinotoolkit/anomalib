"""Anomalib Torch data validators.

This module provides validators for PyTorch tensors used in anomalib. The validators
ensure that input data meets the required format specifications.

The following validators are available:

- Image validators:
    - ``ImageValidator``: Validates single image torch tensors
    - ``ImageBatchValidator``: Validates batches of image torch tensors

- Video validators:
    - ``VideoValidator``: Validates single video frame torch tensors
    - ``VideoBatchValidator``: Validates batches of video frame torch tensors

- Depth validators:
    - ``DepthValidator``: Validates single depth map torch tensors
    - ``DepthBatchValidator``: Validates batches of depth map torch tensors
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
