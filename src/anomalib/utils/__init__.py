"""Utility functions and helpers for anomaly detection.

This module provides various utility functions and helpers for:
    - File downloading and management
    - Metric calculation and evaluation
    - Anomaly map computation and processing
    - Result visualization and plotting

The utilities ensure consistent behavior across the library and provide common
functionality used by multiple components.

Example:
    >>> from anomalib.utils.visualization import ImageVisualizer
    >>> # Create visualizer
    >>> visualizer = ImageVisualizer()
    >>> # Generate visualization
    >>> vis_result = visualizer.visualize(image=img, pred_mask=mask)

The module is organized into submodules for different types of utilities:
    - ``download``: Functions for downloading datasets and models
    - ``metrics``: Implementations of evaluation metrics
    - ``map``: Tools for generating and processing anomaly maps
    - ``visualization``: Classes for visualizing detection results
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .attrs import get_nested_attr

__all__ = ["get_nested_attr"]
