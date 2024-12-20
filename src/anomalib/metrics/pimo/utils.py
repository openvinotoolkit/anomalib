"""Utility functions for PIMO metrics.

This module provides utility functions for working with PIMO (Per-Image Metric
Optimization) metrics in PyTorch.

Example:
    >>> import torch
    >>> masks = torch.zeros(3, 32, 32)  # 3 normal images
    >>> masks[1, 10:20, 10:20] = 1  # Add anomaly to middle image
    >>> classes = images_classes_from_masks(masks)
    >>> classes
    tensor([0, 1, 0])
"""

# Original Code
# https://github.com/jpcbertoldo/aupimo
#
# Modified
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import torch

logger = logging.getLogger(__name__)


def images_classes_from_masks(masks: torch.Tensor) -> torch.Tensor:
    """Deduce binary image classes from ground truth masks.

    Determines if each image contains any anomalous pixels (class 1) or is
    completely normal (class 0).

    Args:
        masks: Binary ground truth masks of shape ``(N, H, W)`` where:
            - ``N``: number of images
            - ``H``: image height
            - ``W``: image width
            Values should be 0 (normal) or 1 (anomalous).

    Returns:
        torch.Tensor: Binary tensor of shape ``(N,)`` containing image-level
        classes where:
            - 0: normal image (no anomalous pixels)
            - 1: anomalous image (contains anomalous pixels)

    Example:
        >>> masks = torch.zeros(3, 32, 32)  # 3 normal images
        >>> masks[1, 10:20, 10:20] = 1  # Add anomaly to middle image
        >>> images_classes_from_masks(masks)
        tensor([0, 1, 0])
    """
    return (masks == 1).any(axis=(1, 2)).to(torch.int32)
