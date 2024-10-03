"""ImageItem visualizer."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from PIL import Image

from anomalib.data import ImageItem
from anomalib.utils.path import convert_to_title_case
from anomalib.visualization.image.functional import (
    add_text_to_image,
    create_image_grid,
    overlay_images,
    visualize_field,
)

logger = logging.getLogger(__name__)

DEFAULT_FIELDS_CONFIG = {
    "image": {},
    "gt_mask": {},
    "pred_mask": {},
    "anomaly_map": {"colormap": True, "normalize": False},
}

DEFAULT_OVERLAY_FIELDS_CONFIG = {
    "gt_mask": {"color": (255, 255, 255), "alpha": 0.5, "mode": "contour"},
    "pred_mask": {"color": (255, 0, 0), "alpha": 0.5, "mode": "contour"},
}

DEFAULT_TEXT_CONFIG = {
    "enable": True,
    "font": None,
    "size": None,
    "color": "white",
    "background": (0, 0, 0, 128),
}


def visualize_image_item(  # noqa: C901 - NOTE: Complexity is 17/15, to be refactored.
    item: ImageItem,
    fields: list[str] | None = None,
    overlay_fields: list[tuple[str, list[str]]] | None = None,
    field_size: tuple[int, int] = (256, 256),
    fields_config: dict[str, dict[str, Any]] = DEFAULT_FIELDS_CONFIG,
    overlay_fields_config: dict[str, dict[str, Any]] = DEFAULT_OVERLAY_FIELDS_CONFIG,
    text_config: dict[str, Any] = DEFAULT_TEXT_CONFIG,
) -> Image.Image | None:
    """Visualize specified fields of an ImageItem with configurable field, overlay, and text options.

    Args:
        item (ImageItem): The ImageItem to visualize.
        fields (list[str] | None): List of fields to visualize individually. Order is preserved in output.
        overlay_fields (list[tuple[str, list[str]]] | None): List of tuples specifying which fields to overlay.
        field_size (tuple[int, int]): Size to resize each field image.
        fields_config (dict[str, dict[str, Any]]): Custom configurations for field visualization.
        overlay_fields_config (dict[str, dict[str, Any]]): Custom configurations for field overlays.
        text_config (dict[str, Any]): Configuration for text overlay.
            Use {"enable": False} to disable text, or specify font options.
            Default is {"enable": True, "font": None, "size": None, "color": "white", "background": (0, 0, 0, 128)}.

    Returns:
        Image.Image | None: The visualized image grid, or None if no valid fields.

    Examples:
        Basic usage with default settings:
        >>> item = ImageItem(...)  # Your ImageItem instance
        >>> visualized = visualize_image_item(item)

        Customizing fields to visualize:
        >>> visualized = visualize_image_item(
        ...     item,
        ...     fields=["image", "gt_mask", "anomaly_map"],
        ...     overlay_fields=[("image", ["anomaly_map"])]
        ... )

        Adjusting field size:
        >>> visualized = visualize_image_item(item, field_size=(512, 512))

        Customizing anomaly map visualization:
        >>> visualized = visualize_image_item(
        ...     item,
        ...     fields_config={
        ...         "anomaly_map": {"colormap": True, "normalize": True}
        ...     }
        ... )

        Modifying overlay appearance:
        >>> visualized = visualize_image_item(
        ...     item,
        ...     overlay_fields_config={
        ...         "pred_mask": {"alpha": 0.7, "color": (255, 0, 0), "mode": "fill"},
        ...         "anomaly_map": {"alpha": 0.5, "color": (0, 255, 0), "mode": "contour"}
        ...     }
        ... )

        Customizing text overlay:
        >>> visualized = visualize_image_item(
        ...     item,
        ...     text_config={
        ...         "font": "arial.ttf",
        ...         "size": 20,
        ...         "color": "yellow",
        ...         "background": (0, 0, 0, 200)
        ...     }
        ... )

        Advanced configuration combining multiple customizations:
        >>> visualized = visualize_image_item(
        ...     item,
        ...     fields=["image", "gt_mask", "anomaly_map", "pred_mask"],
        ...     overlay_fields=[("image", ["anomaly_map"]), ("image", ["pred_mask"])],
        ...     field_size=(384, 384),
        ...     fields_config={
        ...         "anomaly_map": {"colormap": True, "normalize": True},
        ...         "pred_mask": {"color": (0, 0, 255)}
        ...     },
        ...     overlay_fields_config={
        ...         "anomaly_map": {"alpha": 0.6, "mode": "fill"},
        ...         "pred_mask": {"alpha": 0.7, "mode": "contour"}
        ...     },
        ...     text_config={
        ...         "font": "times.ttf",
        ...         "size": 24,
        ...         "color": "white",
        ...         "background": (0, 0, 0, 180)
        ...     }
        ... )
    """
    # Merge default and custom configurations
    fields_config = {**DEFAULT_FIELDS_CONFIG, **(fields_config or {})}
    overlay_fields_config = {**DEFAULT_OVERLAY_FIELDS_CONFIG, **(overlay_fields_config or {})}

    # Merge default and custom text configurations
    text_config = {**DEFAULT_TEXT_CONFIG, **(text_config or {})}
    add_text = text_config.pop("enable", True)

    field_images: dict[str, Image.Image] = {}
    output_images: list[Image.Image] = []

    # Collect all fields that need to be visualized
    all_fields = set(fields or [])
    if overlay_fields:
        for base, overlays in overlay_fields:
            all_fields.add(base)
            all_fields.update(overlays)

    # Visualize all required fields
    for field in all_fields:
        # NOTE: Use ``visualize_field`` for image reading once pre-processing is done in the model.
        if field == "image":
            image = Image.open(item.image_path)
        else:
            value = getattr(item, field)
            field_config = fields_config.get(field, {})
            image = visualize_field(field, value, **field_config)

        if image:
            image = image.resize(field_size)
            field_images[field] = image

    # Process individual fields
    if fields:
        for field in fields:
            if field in field_images:
                image = field_images[field].copy()
                if add_text:
                    title = convert_to_title_case(field)
                    image = add_text_to_image(image, title, **text_config)
                output_images.append(image)

    # Process overlay fields
    if overlay_fields:
        for base, overlays in overlay_fields:
            if base in field_images:
                base_image = field_images[base].copy()
                valid_overlays = [overlay for overlay in overlays if overlay in field_images]

                for overlay in valid_overlays:
                    overlay_field_config = overlay_fields_config.get(overlay, {})
                    base_image = overlay_images(base_image, field_images[overlay], **overlay_field_config)

                if valid_overlays:
                    if add_text:
                        title = (
                            f"{convert_to_title_case(base)} + "
                            f"{'+'.join(convert_to_title_case(o) for o in valid_overlays)}"
                        )
                        base_image = add_text_to_image(base_image, title, **text_config)
                    output_images.append(base_image)

    if not output_images:
        logger.warning("No valid fields to visualize.")
        return None

    return create_image_grid(output_images, nrow=len(output_images))
