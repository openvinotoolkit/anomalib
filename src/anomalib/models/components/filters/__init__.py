"""Filters used by anomaly detection models.

This module provides filter implementations that can be used for image
preprocessing and feature enhancement in anomaly detection models.

Classes:
    GaussianBlur2d: 2D Gaussian blur filter implementation.

Example:
    >>> import torch
    >>> from anomalib.models.components.filters import GaussianBlur2d
    >>> # Create a Gaussian blur filter
    >>> blur = GaussianBlur2d(kernel_size=3, sigma=1.0)
    >>> # Apply blur to input tensor
    >>> input_tensor = torch.randn(1, 3, 256, 256)
    >>> blurred = blur(input_tensor)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .blur import GaussianBlur2d

__all__ = ["GaussianBlur2d"]
