"""ImageItem visualization module.

This module provides utilities for visualizing ``ImageItem`` objects, which contain
images and their associated anomaly detection results. The key components include:

    - Functions for visualizing individual fields (image, masks, anomaly maps)
    - Support for overlaying multiple fields
    - Configurable visualization parameters
    - Text annotation capabilities

Example:
    >>> from anomalib.data import ImageItem
    >>> from anomalib.visualization.image.item_visualizer import visualize_image_item
    >>> # Create an ImageItem
    >>> item = ImageItem(image=img, pred_mask=mask)
    >>> # Generate visualization
    >>> vis_result = visualize_image_item(item)

The module ensures consistent visualization by:
    - Providing standardized field configurations
    - Supporting flexible overlay options
    - Handling text annotations
    - Maintaining consistent output formats

Note:
    All visualization functions preserve the input image format and dimensions
    unless explicitly specified in the configuration.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from PIL import Image

from anomalib.data import ImageItem

logger = logging.getLogger(__name__)

DEFAULT_FIELDS_CONFIG = {
    "image": {},
    "gt_mask": {},
    "pred_mask": {},
    "anomaly_map": {"colormap": True, "normalize": False},
}

DEFAULT_OVERLAY_FIELDS_CONFIG = {
    "gt_mask": {"color": (255, 255, 255), "alpha": 1.0, "mode": "contour"},
    "pred_mask": {"color": (255, 0, 0), "alpha": 1.0, "mode": "contour"},
}

DEFAULT_TEXT_CONFIG = {
    "enable": True,
    "font": None,
    "size": None,
    "color": "white",
    "background": (0, 0, 0, 128),
}


def visualize_image_item(
    item: ImageItem,
    fields: list[str] | None = None,
    overlay_fields: list[tuple[str, list[str]]] | None = None,
    field_size: tuple[int, int] = (256, 256),
    fields_config: dict[str, dict[str, Any]] = DEFAULT_FIELDS_CONFIG,
    overlay_fields_config: dict[str, dict[str, Any]] = DEFAULT_OVERLAY_FIELDS_CONFIG,
    text_config: dict[str, Any] = DEFAULT_TEXT_CONFIG,
) -> Image.Image | None:
    """Visualize specified fields of an ``ImageItem`` with configurable options.

    This function creates visualizations for individual fields and overlays of an
    ``ImageItem``. It supports customization of field visualization, overlay
    composition, and text annotations.

    Args:
        item: An ``ImageItem`` instance containing the data to visualize.
        fields: A list of field names to visualize individually. If ``None``, no
            individual fields are visualized.
        overlay_fields: A list of tuples, each containing a base field and a list
            of fields to overlay on it. If ``None``, no overlays are created.
        field_size: A tuple ``(width, height)`` specifying the size of each
            visualized field.
        fields_config: A dictionary of field-specific visualization
            configurations.
        overlay_fields_config: A dictionary of overlay-specific configurations.
        text_config: A dictionary of text annotation configurations.

    Returns:
        A PIL ``Image`` containing the visualized fields and overlays, or
        ``None`` if no valid fields to visualize.

    Raises:
        AttributeError: If a specified field doesn't exist in the ``ImageItem``.
        ValueError: If an invalid configuration is provided.

    Examples:
        Basic usage with default settings:

        >>> item = ImageItem(
        ...     image_path="image.jpg",
        ...     gt_mask=mask,
        ...     pred_mask=pred,
        ...     anomaly_map=amap
        ... )
        >>> result = visualize_image_item(
        ...     item,
        ...     fields=["image", "gt_mask", "pred_mask", "anomaly_map"]
        ... )

        Visualizing specific fields:

        >>> result = visualize_image_item(item, fields=["image", "anomaly_map"])

        Creating an overlay:

        >>> result = visualize_image_item(
        ...     item,
        ...     fields=["image"],
        ...     overlay_fields=[("image", ["anomaly_map"])]
        ... )

        Multiple overlays:

        >>> result = visualize_image_item(
        ...     item,
        ...     overlay_fields=[
        ...         ("image", ["gt_mask"]),
        ...         ("image", ["pred_mask"]),
        ...         ("image", ["anomaly_map"])
        ...     ]
        ... )

        Customizing field visualization:

        >>> result = visualize_image_item(
        ...     item,
        ...     fields=["image", "anomaly_map"],
        ...     fields_config={
        ...         "anomaly_map": {"colormap": True, "normalize": True}
        ...     }
        ... )

        Adjusting overlay transparency:

        >>> result = visualize_image_item(
        ...     item,
        ...     overlay_fields=[("image", ["gt_mask", "pred_mask"])],
        ...     overlay_fields_config={
        ...         "gt_mask": {"alpha": 0.3},
        ...         "pred_mask": {"alpha": 0.7}
        ...     }
        ... )

        Customizing text annotations:

        >>> result = visualize_image_item(
        ...     item,
        ...     fields=["image", "gt_mask"],
        ...     text_config={
        ...         "font": "arial.ttf",
        ...         "size": 20,
        ...         "color": "yellow",
        ...         "background": (0, 0, 0, 180)
        ...     }
        ... )

        Disabling text annotations:

        >>> result = visualize_image_item(
        ...     item,
        ...     fields=["image", "gt_mask"],
        ...     text_config={"enable": False}
        ... )

        Combining multiple customizations:

        >>> result = visualize_image_item(
        ...     item,
        ...     fields=["image", "gt_mask", "pred_mask"],
        ...     overlay_fields=[("image", ["anomaly_map"])],
        ...     field_size=(384, 384),
        ...     fields_config={
        ...         "anomaly_map": {"colormap": True, "normalize": True},
        ...     },
        ...     overlay_fields_config={
        ...         "anomaly_map": {"colormap": True},
        ...     },
        ...     text_config={
        ...         "font": "times.ttf",
        ...         "size": 24,
        ...         "color": "white",
        ...         "background": (0, 0, 0, 200)
        ...     }
        ... )

        Handling missing fields gracefully:

        >>> item_no_pred = ImageItem(
        ...     image_path="image.jpg",
        ...     gt_mask=mask,
        ...     anomaly_map=amap
        ... )
        >>> result = visualize_image_item(
        ...     item_no_pred,
        ...     fields=["image", "gt_mask", "pred_mask", "anomaly_map"]
        ... )
        # This will visualize all available fields, skipping 'pred_mask'

        Custom ordering of fields and overlays:

        >>> result = visualize_image_item(
        ...     item,
        ...     fields=["anomaly_map", "image", "gt_mask"],
        ...     overlay_fields=[
        ...         ("image", ["pred_mask"]),
        ...         ("image", ["gt_mask", "anomaly_map"]),
        ...     ]
        ... )
        # This will maintain the specified order in the output

        Different masking strategies:

        1. Binary mask visualization:

        >>> result = visualize_image_item(
        ...     item,
        ...     fields=["gt_mask", "pred_mask"],
        ...     fields_config={
        ...         "gt_mask": {"mode": "binary"},
        ...         "pred_mask": {"mode": "binary"}
        ...     }
        ... )

        2. Contour mask visualization:

        >>> result = visualize_image_item(
        ...     item,
        ...     fields=["gt_mask", "pred_mask"],
        ...     fields_config={
        ...         "gt_mask": {"mode": "contour", "color": (0, 255, 0)},
        ...         "pred_mask": {"mode": "contour", "color": (255, 0, 0)}
        ...     }
        ... )

        3. Filled mask visualization:

        >>> result = visualize_image_item(
        ...     item,
        ...     fields=["gt_mask", "pred_mask"],
        ...     fields_config={
        ...         "gt_mask": {"mode": "fill", "color": (0, 255, 0), "alpha": 0.5},
        ...         "pred_mask": {"mode": "fill", "color": (255, 0, 0), "alpha": 0.5}
        ...     }
        ... )

        4. Mixed masking strategies:

        >>> result = visualize_image_item(
        ...     item,
        ...     fields=["image"],
        ...     overlay_fields=[("image", ["gt_mask", "pred_mask"])],
        ...     overlay_fields_config={
        ...         "gt_mask": {"mode": "contour", "color": (0, 255, 0), "alpha": 0.7},
        ...         "pred_mask": {"mode": "fill", "color": (255, 0, 0), "alpha": 0.3}
        ...     }
        ... )

        5. Combining masking strategies with anomaly map:

        >>> result = visualize_image_item(
        ...     item,
        ...     fields=["image", "anomaly_map"],
        ...     overlay_fields=[("image", ["gt_mask", "pred_mask"])],
        ...     fields_config={
        ...         "anomaly_map": {"colormap": True, "normalize": True}
        ...     },
        ...     overlay_fields_config={
        ...         "gt_mask": {"mode": "contour", "color": (0, 255, 0), "alpha": 0.7},
        ...         "pred_mask": {"mode": "fill", "color": (255, 0, 0), "alpha": 0.3}
        ...     }
        ... )

    Note:
        - The function preserves the order of fields as specified in the input.
        - If a field is not available in the ``ImageItem``, it will be skipped
          without raising an error.
        - The function uses default configurations if not provided, which can be
          overridden by passing custom configurations.
        - For mask visualization, the ``mode`` parameter in ``fields_config`` or
          ``overlay_fields_config`` determines how the mask is displayed:

          * ``'binary'``: Shows the mask as a black and white image
          * ``'contour'``: Displays only the contours of the mask
          * ``'fill'``: Fills the mask area with a specified color and
            transparency
    """
