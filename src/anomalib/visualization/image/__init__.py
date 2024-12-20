"""Image visualization module for anomaly detection.

This module provides utilities for visualizing images and anomaly detection results.
The key components include:

    - Functions for visualizing anomaly maps and segmentation masks
    - Tools for overlaying images and adding text annotations
    - Colormap application utilities
    - Image item visualization
    - ``ImageVisualizer`` class for consistent visualization

Example:
    >>> from anomalib.visualization.image import ImageVisualizer
    >>> # Create visualizer
    >>> visualizer = ImageVisualizer()
    >>> # Generate visualization
    >>> vis_result = visualizer.visualize(image=img, pred_mask=mask)

The module ensures consistent visualization by:
    - Providing standardized colormaps and overlays
    - Supporting both classification and segmentation results
    - Handling various input formats
    - Maintaining consistent output formats

Note:
    All visualization functions preserve the input image format and dimensions
    unless explicitly specified otherwise.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .functional import add_text_to_image, apply_colormap, overlay_images, visualize_anomaly_map, visualize_mask
from .item_visualizer import visualize_image_item
from .visualizer import ImageVisualizer

__all__ = [
    # Visualization functions
    "add_text_to_image",
    "apply_colormap",
    "overlay_images",
    "visualize_anomaly_map",
    "visualize_mask",
    # Visualize ImageItem
    "visualize_image_item",
    # Visualization class
    "ImageVisualizer",
]
