"""Computer vision utilities for anomaly detection.

This module provides computer vision utilities used by the anomalib library for
processing and analyzing images during anomaly detection.

The utilities include:
    - Connected components analysis for both CPU and GPU
    - Image processing operations
    - Computer vision helper functions

Example:
    >>> from anomalib.utils.cv import connected_components_cpu
    >>> # Process image to get binary mask
    >>> mask = get_binary_mask(image)
    >>> # Find connected components
    >>> labels = connected_components_cpu(mask)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .connected_components import connected_components_cpu, connected_components_gpu

__all__ = ["connected_components_cpu", "connected_components_gpu"]
