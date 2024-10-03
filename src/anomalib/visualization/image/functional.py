"""Visualizer for ImageItem fields using PIL and torchvision."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
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


def draw_mask_contours(
    image: Image.Image,
    mask: Image.Image,
    color: tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """Draw contours of a mask on an input image using PIL.

    Args:
        image (Image.Image): The base image on which to draw contours.
        mask (Image.Image): The mask image used to find contours.
        color (tuple[int, int, int], optional): RGB color for the contours. Defaults to (255, 0, 0) (red).

    Returns:
        Image.Image: A new image with mask contours drawn on it.

    Examples:
        >>> from PIL import Image, ImageDraw

        >>> input_img = Image.new('RGB', (100, 100), color='white')

        >>> mask_img = Image.new('L', (100, 100), color=0)
        >>> draw = ImageDraw.Draw(mask_img)
        >>> draw.rectangle([25, 25, 75, 75], fill=255)

        >>> result = draw_contours(input_img, mask_img, color=(0, 255, 0))
        >>> result.show()
    """
    # Ensure mask is binary
    mask = mask.convert("L")

    # Find edges of the mask
    edges = mask.filter(ImageFilter.FIND_EDGES)

    # Create a colored version of the edges
    colored_edges = Image.new("RGB", image.size, color)
    colored_edges.putalpha(edges.convert("L"))

    # Composite the colored edges onto the input image
    return Image.alpha_composite(image.convert("RGBA"), colored_edges)


def fill_mask_area(
    image: Image.Image,
    mask: Image.Image,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
) -> Image.Image:
    """Fill a mask area on an image with a color.

    This function takes an input image and a mask, converts them to the appropriate
    modes, adjusts the alpha of the mask, and then composites a color onto the image
    using the adjusted mask.

    Args:
        image (Image.Image): The input image to overlay the mask on.
        mask (Image.Image): The mask to be overlaid on the image.
        color (tuple[int, int, int], optional): The color of the overlay. Defaults to (255, 0, 0).
        alpha (float, optional): The alpha value for the overlay. Defaults to 0.5.

    Returns:
        Image.Image: A new image with the mask overlaid on the input image.

    Examples:
        >>> from PIL import Image, ImageDraw

        >>> # Create a sample input image (white background)
        >>> input_img = Image.new('RGB', (100, 100), color='white')
        >>> # Create a sample mask (a rectangle)
        >>> mask_img = Image.new('L', (100, 100), color=0)
        >>> draw = ImageDraw.Draw(mask_img)
        >>> draw.rectangle([25, 25, 75, 75], fill=255)
        >>> result = overlay_mask(input_img, mask_img, alpha=0.3)
        >>> result.show()  # Display the result

        # To compare original and result side by side:
        >>> comparison = Image.new('RGB', (input_img.width * 2, input_img.height))
        >>> comparison.paste(input_img, (0, 0))
        >>> comparison.paste(result, (input_img.width, 0))
        >>> comparison.show()  # Display the comparison
    """
    # Ensure input image is in RGBA mode
    image = image.convert("RGBA")

    # Ensure mask is in grayscale mode
    mask = mask.convert("L")

    # Adjust the alpha of the mask
    alpha_mask = Image.new("L", mask.size, int(255 * alpha))
    adjusted_mask = ImageChops.multiply(mask, alpha_mask)

    # Create a solid color image for the overlay
    overlay = Image.new("RGBA", image.size, color)

    # Composite the overlay onto the input using the adjusted mask
    return Image.composite(overlay, image, adjusted_mask)


def overlay_mask(
    image: Image.Image,
    mask: Image.Image,
    alpha: float = 0.2,
    color: tuple[int, int, int] = (255, 0, 0),
    mode: Literal["contour", "fill"] = "contour",
) -> Image.Image:
    """Overlay a mask on an image.

    Args:
        image (Image.Image): The base image.
        mask (Image.Image): The mask to overlay.
        alpha (float): The alpha value for blending (0.0 to 1.0).
        color (tuple[int, int, int]): The color of the overlay.
        mode (Literal["contour", "fill"]): The mode of the overlay.
            ``contour`` draws contours on the mask.
            ``fill`` fills the mask area with color.
            Defaults to ``contour``.

    Returns:
        Image.Image: The overlaid image.

    Examples:
        >>> from PIL import Image, ImageDraw

        >>> input_img = Image.new('RGB', (100, 100), color='white')
        >>> mask_img = Image.new('L', (100, 100), color=0)
        >>> draw = ImageDraw.Draw(mask_img)
        >>> draw.rectangle([25, 25, 75, 75], fill=255)

        >>> result = overlay_mask(input_img, mask_img, mode="contour")
        >>> result.show()

        >>> result = overlay_mask(input_img, mask_img, mode="fill")
        >>> result.show()
    """
    if mode == "contour":
        return draw_mask_contours(image, mask, color=color)
    if mode == "fill":
        return fill_mask_area(image, mask, color=color, alpha=alpha)

    msg = f"Invalid overlay mode: {mode}. Allowed modes are 'contour' or 'fill'."
    raise ValueError(msg)


def overlay_images(
    base: Image.Image,
    overlay: Image.Image | list[Image.Image],
    *,  # Mark the following arguments as keyword-only
    alpha: float = 0.2,
    color: tuple[int, int, int] = (255, 0, 0),
    mode: Literal["contour", "fill"] = "contour",
) -> Image.Image:
    """Overlay multiple images on top of a base image with a specified alpha value.

    If the overlay is a mask (L mode), draw its contours on the image instead.

    Args:
        base: The base PIL Image.
        overlay: PIL Image or list of PIL Images to overlay on top of the base image.
        alpha: The alpha value for blending (0.0 to 1.0).
        color: Contour color for the mask. If None, the mask is returned as is.
        mode: Mode of the overlay mask. ``contour`` draws contours on the mask.
            ``fill`` fills the mask area with color.
            Defaults to ``contour``.

    Returns:
        A new PIL Image with all overlays applied.
    """
    # Ensure base image is in RGB mode
    base = base.convert("RGB")

    if not isinstance(overlay, list):
        overlay = [overlay]

    for ov in overlay:
        if ov.mode == "L":
            # L modes are masks, so we can draw contours or overlay them.
            base = overlay_mask(base, ov, alpha=alpha, color=color, mode=mode)
        else:
            # Ensure overlay image is in RGB mode and resize if necessary
            colored_overlay = ov.convert("RGB")
            if base.size != colored_overlay.size:
                colored_overlay = colored_overlay.resize(base.size)

            # Blend the overlay with the base image
            base = Image.blend(base, colored_overlay, alpha)

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
    color: tuple[int, int, int] | None = None,
) -> Image.Image:
    """Visualize a mask by applying a color to it.

    Args:
        mask (Image.Image | torch.Tensor): The input mask. Can be a PIL Image or a PyTorch tensor.
        color (tuple[int, int, int], optional): The color to apply to the mask. If None, the mask is returned as is.

    Returns:
        Image.Image: The visualized mask as a PIL Image.

    Raises:
        TypeError: If the mask is not a PIL Image or PyTorch tensor.

    Examples:
        1. Visualize a PIL Image mask:
        >>> from PIL import Image
        >>> import numpy as np
        >>> mask_array = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
        >>> mask_image = Image.fromarray(mask_array)
        >>> colored_mask = visualize_mask(mask_image, color=(255, 0, 0))  # Red mask
        >>> colored_mask.show()

        2. Visualize a PyTorch tensor mask:
        >>> import torch
        >>> mask_tensor = torch.randint(0, 2, size=(100, 100), dtype=torch.bool)
        >>> colored_mask = visualize_mask(mask_tensor, color=(0, 255, 0))  # Green mask
        >>> colored_mask.show()

        3. Visualize a grayscale PyTorch tensor mask:
        >>> mask_tensor = torch.randint(0, 256, size=(100, 100), dtype=torch.uint8)
        >>> colored_mask = visualize_mask(mask_tensor, color=(0, 0, 255))  # Blue mask
        >>> colored_mask.show()

        4. Visualize without applying color:
        >>> mask_array = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
        >>> mask_image = Image.fromarray(mask_array)
        >>> grayscale_mask = visualize_mask(mask_image)
        >>> grayscale_mask.show()
    """
    if isinstance(mask, torch.Tensor):
        if mask.dtype == torch.bool:
            mask = mask.to(torch.uint8) * 255
        mask = to_pil_image(mask)

    if not isinstance(mask, Image.Image):
        msg = "Mask must be a PIL Image"
        raise TypeError(msg)

    if color:
        # Create a solid color image
        solid_color = Image.new("RGB", mask.size, color)

        # Use the original mask as alpha
        mask = Image.composite(solid_color, Image.new("RGB", mask.size, (0, 0, 0)), mask)

    return mask


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


def visualize_field(
    field: str,
    value: torch.Tensor,
    *,  # Mark the following arguments as keyword-only
    colormap: bool = True,
    normalize: bool = False,
) -> Image.Image | None:
    """Visualize a single field of an ImageItem."""
    if field == "image":
        msg = "Image visualization is not implemented yet"
        raise NotImplementedError(msg)
    if field in {"gt_mask", "pred_mask"}:
        image = visualize_mask(value)
    if field == "anomaly_map":
        image = visualize_anomaly_map(value, normalize=normalize, colormap=colormap)

    return image
