"""Visualizer for ImageItem fields using PIL and torchvision."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect
import logging
import sys
from collections.abc import Callable
from typing import Any, Literal

import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from torchvision.transforms.functional import to_pil_image

logger = logging.getLogger(__name__)


def dynamic_font_size(image_size: tuple[int, int], min_size: int = 20, max_size: int = 100, divisor: int = 10) -> int:
    """Calculate a dynamic font size based on image dimensions.

    Args:
        image_size: Tuple of image dimensions (width, height).
        min_size: Minimum font size (default: 20).
        max_size: Maximum font size (default: 100).
        divisor: Divisor for calculating font size (default: 10).

    Returns:
        Calculated font size within the specified range.
    """
    min_dimension = min(image_size)
    return max(min_size, min(max_size, min_dimension // divisor))


def add_text_to_image(
    image: Image.Image,
    text: str,
    font: str | None = None,
    size: int | None = None,
    color: tuple[int, int, int] | str = "white",
    background: tuple[int, ...] | str | None = (0, 0, 0, 128),  # Default to semi-transparent black
    position: tuple[int, int] = (10, 10),
    padding: int = 3,
) -> Image.Image:
    """Add text to an image with configurable parameters."""
    # Create a new RGBA image as a transparent overlay
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if size is None:
        size = dynamic_font_size(image.size)

    try:
        image_font = ImageFont.truetype(font, size) if font else ImageFont.load_default()
    except OSError:
        logger.warning(f"Failed to load font '{font}'. Using default font.")
        image_font = ImageFont.load_default()

    # Calculate text size and position
    text_bbox = draw.textbbox(position, text, font=image_font)
    text_position = position
    background_bbox = (text_bbox[0] - padding, text_bbox[1] - padding, text_bbox[2] + padding, text_bbox[3] + padding)

    # Draw background if specified
    if background is not None:
        draw.rectangle(background_bbox, fill=background)

    # Draw text
    draw.text(text_position, text, font=image_font, fill=color)

    # Composite the overlay onto the original image
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def apply_colormap(image: Image.Image) -> Image.Image:
    """Apply a colormap to a single-channel PIL Image using torch and PIL.

    This function converts a grayscale image to a colored image using the 'jet' colormap.

    Args:
        image (Image.Image): A single-channel PIL Image or an object that can be converted to PIL Image.

    Returns:
        Image.Image: A new PIL Image with the colormap applied.

    Raises:
        TypeError: If the input cannot be converted to a PIL Image.

    Example:
        >>> from PIL import Image
        >>> import numpy as np
        >>> # Create a sample grayscale image
        >>> gray_image = Image.fromarray(np.random.randint(0, 256, (100, 100), dtype=np.uint8), mode='L')
        >>> # Apply the jet colormap
        >>> colored_image = apply_colormap(gray_image)
        >>> colored_image.show()
    """
    # Try to convert the input to a PIL Image if it's not already
    if not isinstance(image, Image.Image):
        try:
            image = Image.fromarray(image)
        except TypeError:
            msg = "Input must be a PIL Image object or an object that can be converted to PIL Image"
            raise TypeError(msg) from None

    # Ensure image is in 'L' mode (8-bit pixels, black and white)
    if image.mode != "L":
        image = image.convert("L")

    # Define colormap values for the 'jet' colormap
    colormap_values = [
        (0, 0, 143),  # Dark blue
        (0, 0, 255),  # Blue
        (0, 127, 255),  # Light blue
        (0, 255, 255),  # Cyan
        (127, 255, 127),  # Light green
        (255, 255, 0),  # Yellow
        (255, 127, 0),  # Orange
        (255, 0, 0),  # Red
        (127, 0, 0),  # Dark red
    ]

    # Create a linear interpolation of the colormap
    colormap_tensor = torch.tensor(colormap_values, dtype=torch.float32)
    colormap_tensor = colormap_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

    # Interpolate to create a smooth 256-color palette
    interpolated = F.interpolate(colormap_tensor, size=(256, 3), mode="bilinear", align_corners=False)
    interpolated = interpolated.squeeze().byte()

    # Convert the interpolated tensor to a flat list for PIL
    palette = interpolated.flatten().tolist()

    # Apply the colormap to the image
    colored_image = image.convert("P")  # Convert to 8-bit pixels, mapped to a palette
    colored_image.putpalette(palette)  # Apply our custom palette
    return colored_image.convert("RGB")  # Convert back to RGB for display


def overlay_image(base: Image.Image, overlay: Image.Image, alpha: float = 0.5) -> Image.Image:
    """Overlay an image on top of another image with a specified alpha value.

    Args:
        base (Image.Image): The base image.
        overlay (Image.Image): The image to overlay.
        alpha (float): The alpha value for blending (0.0 to 1.0). Defaults to 0.5.

    Returns:
        Image.Image: The image with the overlay applied.

    Examples:
        # Overlay a random mask on an image
        >>> from PIL import Image, ImageDraw

        >>> image = Image.new('RGB', (200, 200), color='green')
        >>> draw = ImageDraw.Draw(image)
        >>> draw.polygon([(50, 50), (150, 50), (100, 150)], fill='yellow')

        >>> mask = Image.new('L', (200, 200), color=0)
        >>> draw = ImageDraw.Draw(mask)
        >>> draw.rectangle([75, 75, 125, 125], fill=255)

        >>> result = overlay_image(image, mask, alpha=0.3)
        >>> result.show()
    """
    base = base.convert("RGBA")
    overlay = overlay.convert("RGBA")

    # Resize mask to match input image size if necessary
    if base.size != overlay.size:
        overlay = overlay.resize(base.size)

    # Adjust the alpha of the mask
    alpha_mask = overlay.split()[3]
    alpha_mask = ImageEnhance.Brightness(alpha_mask).enhance(alpha)
    overlay.putalpha(alpha_mask)

    # Composite the mask over the input image
    return Image.alpha_composite(base, overlay)


def overlay_images(
    base: Image.Image,
    overlays: Image.Image | list[Image.Image],
    alpha: float | list[float] = 0.5,
) -> Image.Image:
    """Overlay multiple images on top of a base image with a specified alpha value.

    If the overlay is a mask (L mode), draw its contours on the image instead.

    Args:
        base: The base PIL Image.
        overlays: PIL Image or list of PIL Images to overlay on top of the base image.
        alpha: The alpha value for blending (0.0 to 1.0). Defaults to 0.5.

    Returns:
        A new PIL Image with all overlays applied.

    Examples:
        # Overlay a single image
        >>> from PIL import Image, ImageDraw
        >>> image = Image.new('RGB', (200, 200), color='green')
        >>> draw = ImageDraw.Draw(image)
        >>> draw.polygon([(50, 50), (150, 50), (100, 150)], fill='yellow')

        >>> mask = Image.new('L', (200, 200), color=0)
        >>> draw = ImageDraw.Draw(mask)
        >>> draw.rectangle([75, 75, 125, 125], fill=255)

        >>> result = overlay_images(image, mask)

        # Overlay multiple images
        >>> image = Image.new('RGB', (200, 200), color='green')
        >>> draw = ImageDraw.Draw(image)
        >>> draw.polygon([(50, 50), (150, 50), (100, 150)], fill='yellow')

        >>> mask1 = Image.new('L', (200, 200), color=0)
        >>> draw = ImageDraw.Draw(mask1)
        >>> draw.rectangle([25, 25, 75, 75], fill=255)

        >>> mask2 = Image.new('L', (200, 200), color=0)
        >>> draw = ImageDraw.Draw(mask2)
        >>> draw.ellipse([50, 50, 150, 100], fill=255)

        >>> result = overlay_images(image, [mask1, mask2])
    """
    if not isinstance(overlays, list):
        overlays = [overlays]

    if not isinstance(alpha, list):
        alphas = [alpha]

    for overlay, overlay_alpha in zip(overlays, alphas, strict=False):
        base = overlay_image(base, overlay, alpha=overlay_alpha)

    return base


def visualize_anomaly_map(
    anomaly_map: Image.Image | torch.Tensor,
    colormap: bool = True,
    normalize: bool = False,
) -> Image.Image:
    """Visualize the anomaly map.

    This function takes an anomaly map as input and applies normalization and/or colormap
    based on the provided parameters.

    Args:
        anomaly_map (Image.Image | torch.Tensor): The input anomaly map as a PIL Image or torch Tensor.
        colormap (bool, optional): Whether to apply a colormap to the anomaly map. Defaults to True.
        normalize (bool, optional): Whether to normalize the anomaly map. Defaults to False.

    Returns:
        Image.Image: The visualized anomaly map as a PIL Image in RGB mode.

    Example:
        >>> from PIL import Image
        >>> import numpy as np
        >>> import torch

        >>> # Create a sample anomaly map as PIL Image
        >>> anomaly_map_pil = Image.fromarray(np.random.rand(100, 100).astype(np.float32), mode='F')

        >>> # Create a sample anomaly map as torch Tensor
        >>> anomaly_map_tensor = torch.rand(100, 100)

        >>> # Visualize the anomaly maps
        >>> visualized_map_pil = visualize_anomaly_map(anomaly_map_pil, normalize=True, colormap=True)
        >>> visualized_map_tensor = visualize_anomaly_map(anomaly_map_tensor, normalize=True, colormap=True)
        >>> visualized_map_pil.show()
        >>> visualized_map_tensor.show()
    """
    image = to_pil_image(anomaly_map) if isinstance(anomaly_map, torch.Tensor) else anomaly_map.copy()

    if normalize:
        # Get the min and max pixel values
        min_value = image.getextrema()[0]
        max_value = image.getextrema()[1]

        # Create a normalized image
        image = image.point(lambda x: (x - min_value) * 255 / (max_value - min_value))

    return apply_colormap(image) if colormap else image.convert("RGB")


def visualize_mask(
    mask: Image.Image | torch.Tensor,
    *,
    mode: Literal["contour", "fill", "binary", "L", "1"] = "binary",
    alpha: float = 0.5,
    color: tuple[int, int, int] = (255, 0, 0),
    background_color: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Image.Image:
    """Visualize a mask with different modes.

    Args:
        mask (Image.Image | torch.Tensor): The input mask. Can be a PIL Image or a PyTorch tensor.
        mode (Literal["contour", "binary", "fill"]): The visualization mode.
            - "contour": Draw contours of the mask.
            - "fill": Fill the masked area with a color.
            - "binary": Return the original binary mask.
            - "L": Return the original grayscale mask.
            - "1": Return the original binary mask.
        alpha (float): The alpha value for blending (0.0 to 1.0). Only used in "fill" mode.
            Defaults to 0.5.
        color (tuple[int, int, int]): The color to apply to the mask.
            Defaults to (255, 0, 0) (red).
        background_color (tuple[int, int, int, int]): The background color (RGBA).
            Defaults to (0, 0, 0, 0) (transparent).

    Returns:
        Image.Image: The visualized mask as a PIL Image.

    Raises:
        TypeError: If the mask is not a PIL Image or PyTorch tensor.
        ValueError: If an invalid mode is provided.

    Examples:
        >>> mask_array = np.random.randint(0, 2, size=(100, 100), dtype=np.uint8) * 255
        >>> mask_image = Image.fromarray(mask_array, mode='L')

        >>> contour_mask = visualize_mask(mask_image, mode="contour", color=(255, 0, 0))
        >>> contour_mask.show()

        >>> binary_mask = visualize_mask(mask_image, mode="binary")
        >>> binary_mask.show()

        >>> fill_mask = visualize_mask(mask_image, mode="fill", color=(0, 255, 0), alpha=0.3)
        >>> fill_mask.show()
    """
    # Convert torch.Tensor to PIL Image if necessary
    if isinstance(mask, torch.Tensor):
        if mask.dtype == torch.bool:
            mask = mask.to(torch.uint8) * 255
        mask = to_pil_image(mask)

    if not isinstance(mask, Image.Image):
        msg = "Mask must be a PIL Image or PyTorch tensor"
        raise TypeError(msg)

    # Ensure mask is in binary mode
    mask = mask.convert("L")
    if mode in {"binary", "L", "1"}:
        return mask

    # Create a background image
    background = Image.new("RGBA", mask.size, background_color)

    match mode:
        case "contour":
            # Find edges of the mask
            edges = mask.filter(ImageFilter.FIND_EDGES)

            # Create a colored version of the edges
            colored_edges = Image.new("RGBA", mask.size, (*color, 255))
            colored_edges.putalpha(edges)

            # Composite the colored edges onto the background
            return Image.alpha_composite(background, colored_edges)

        case "fill":
            # Create a solid color image for the overlay
            overlay = Image.new("RGBA", mask.size, (*color, int(255 * alpha)))

            # Use the mask to blend the overlay with the background
            return Image.composite(overlay, background, mask)

        case _:
            msg = f"Invalid mode: {mode}. Allowed modes are 'contour', 'binary', or 'fill'."
            raise ValueError(msg)


def visualize_gt_mask(
    mask: Image.Image | torch.Tensor,
    *,
    mode: Literal["contour", "fill", "binary", "L", "1"] = "binary",
    alpha: float = 0.5,
    color: tuple[int, int, int] = (255, 0, 0),
    background_color: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Image.Image:
    """Visualize a ground truth mask."""
    return visualize_mask(mask, mode=mode, alpha=alpha, color=color, background_color=background_color)


def visualize_pred_mask(
    mask: Image.Image | torch.Tensor,
    *,
    mode: Literal["contour", "fill", "binary", "L", "1"] = "binary",
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
    background_color: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Image.Image:
    """Visualize a prediction mask."""
    return visualize_mask(mask, mode=mode, alpha=alpha, color=color, background_color=background_color)


def create_image_grid(images: list[Image.Image], nrow: int) -> Image.Image:
    """Create a grid of images using PIL.

    Args:
        images: List of PIL Images to arrange in a grid.
        nrow: Number of images per row.

    Returns:
        A new PIL Image containing the grid of images.
    """
    if not images:
        msg = "No images provided to create grid"
        raise ValueError(msg)

    # Assuming all images have the same size
    img_width, img_height = images[0].size

    # Calculate grid dimensions
    ncol = (len(images) + nrow - 1) // nrow  # Ceiling division
    grid_width = nrow * img_width
    grid_height = ncol * img_height

    # Create a new image with white background
    grid_image = Image.new("RGB", (grid_width, grid_height), color="white")

    # Paste images into grid
    for idx, img in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        grid_image.paste(img, (col * img_width, row * img_height))

    return grid_image


def get_field_kwargs(field: str) -> dict[str, Any]:
    """Get the keyword arguments for a visualization function.

    This function retrieves the default keyword arguments for a given visualization function.

    Args:
        field (str): The name of the visualization field (e.g., 'mask', 'anomaly_map').

    Returns:
        dict[str, Any]: A dictionary containing the default keyword arguments for the visualization function.

    Raises:
        ValueError: If the specified field does not have a corresponding visualization function.

    Examples:
        >>> # Get keyword arguments for visualizing a mask
        >>> mask_kwargs = get_field_kwargs('mask')
        >>> print(mask_kwargs)
        {'mode': 'binary', 'color': (255, 0, 0), 'alpha': 0.5, 'background_color': (0, 0, 0, 0)}

        >>> # Get keyword arguments for visualizing an anomaly map
        >>> anomaly_map_kwargs = get_field_kwargs('anomaly_map')
        >>> print(anomaly_map_kwargs)
        {'colormap': True, 'normalize': False}

        >>> # Attempt to get keyword arguments for an invalid field
        >>> get_field_kwargs('invalid_field')
        Traceback (most recent call last):
            ...
        ValueError: 'invalid_field' is not a valid function in the current module.
    """
    # Get the current module
    current_module = sys.modules[__name__]

    # Try to get the function from the current module
    func_name = f"visualize_{field}"
    func = getattr(current_module, func_name, None)

    if func is None or not callable(func):
        msg = f"'{field}' is not a valid function in the current module."
        raise ValueError(msg)

    # Get the signature of the function
    signature = inspect.signature(func)

    # Initialize a dictionary to store keyword argument information
    kwargs = {}

    # Iterate through the parameters
    for name, param in signature.parameters.items():
        # Check if the parameter is a keyword argument
        if param.kind in {inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}:
            if param.default != inspect.Parameter.empty:
                kwargs[name] = param.default
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            kwargs[name] = "Variable keyword arguments (**kwargs)"

    return kwargs


def get_visualize_function(field: str) -> Callable:
    """Get the visualization function for a given field.

    Args:
        field (str): The name of the visualization field
            (e.g., 'image', 'mask', 'anomaly_map').

    Returns:
        Callable: The visualization function corresponding to the given field.

    Raises:
        AttributeError: If the specified field does not have a corresponding
            visualization function.

    Examples:
        >>> from PIL import Image

        Get the visualize function for an anomaly map
        >>> visualize_func = get_visualize_function('anomaly_map')
        >>> anomaly_map = Image.new('F', (256, 256))
        >>> visualized_map = visualize_func(anomaly_map, colormap=True, normalize=True)
        >>> isinstance(visualized_map, Image.Image)
        True

        >>> visualize_func = get_visualize_function('mask')
        >>> mask = Image.new('1', (256, 256))
        >>> visualized_mask = visualize_func(mask, color=(255, 0, 0))
        >>> isinstance(visualized_mask, Image.Image)
        True

        Attempt to get a function for an invalid field
        >>> get_visualize_function('invalid_field')
        Raises AttributeError: module 'anomalib.visualization.image.functional'
        has no attribute 'visualize_invalid_field'
    """
    current_module = sys.modules[__name__]
    func_name = f"visualize_{field}"
    return getattr(current_module, func_name)
