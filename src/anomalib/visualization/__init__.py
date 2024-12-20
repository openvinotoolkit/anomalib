"""Visualization module for anomaly detection.

This module provides utilities for visualizing anomaly detection results. The key
components include:

    - Base ``Visualizer`` class defining the visualization interface
    - ``ImageVisualizer`` class for image-based visualization
    - Functions for visualizing anomaly maps and segmentation masks
    - Tools for visualizing ``ImageItem`` objects

Example:
    >>> from anomalib.visualization import ImageVisualizer
    >>> # Create visualizer
    >>> visualizer = ImageVisualizer()
    >>> # Generate visualization
    >>> vis_result = visualizer.visualize(image=img, pred_mask=mask)

The module ensures consistent visualization by:
    - Providing standardized visualization interfaces
    - Supporting both classification and segmentation results
    - Handling various input formats
    - Maintaining consistent output formats

Note:
    All visualization functions preserve the input format and dimensions unless
    explicitly specified otherwise.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .base import Visualizer
from .image import ImageVisualizer, visualize_anomaly_map, visualize_mask
from .image.item_visualizer import visualize_image_item

__all__ = [
    # Base visualizer class
    "Visualizer",
    # Image visualizer class
    "ImageVisualizer",
    # Image visualization functions
    "visualize_anomaly_map",
    "visualize_image_item",
    "visualize_mask",
]
