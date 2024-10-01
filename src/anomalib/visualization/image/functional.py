"""Visualizer for ImageItem fields using PIL and torchvision."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging

import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image

from anomalib.data import ImageItem
from anomalib.utils.path import convert_to_title_case

# Set up logging
logging.basicConfig(level=logging.INFO)
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
    font_type: str | None = None,
    font_size: int | None = None,
    text_color: tuple[int, int, int] | str = "white",
    background_color: tuple[int, ...] | str | None = (0, 0, 0, 128),  # Default to semi-transparent black
    position: tuple[int, int] = (10, 10),
    padding: int = 3,
) -> Image.Image:
    """Add text to an image with configurable parameters."""
    # Create a new RGBA image as a transparent overlay
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if font_size is None:
        font_size = dynamic_font_size(image.size)

    try:
        font = ImageFont.truetype(font_type, font_size) if font_type else ImageFont.load_default()
    except OSError:
        logger.warning(f"Failed to load font '{font_type}'. Using default font.")
        font = ImageFont.load_default()

    # Calculate text size and position
    text_bbox = draw.textbbox(position, text, font=font)
    text_position = position
    background_bbox = (text_bbox[0] - padding, text_bbox[1] - padding, text_bbox[2] + padding, text_bbox[3] + padding)

    # Draw background if specified
    if background_color is not None:
        draw.rectangle(background_bbox, fill=background_color)

    # Draw text
    draw.text(text_position, text, font=font, fill=text_color)

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


def overlay_images(
    base: Image.Image,
    overlay: Image.Image | list[Image.Image],
    alpha: float,
) -> Image.Image:
    """Overlay multiple images on top of a base image with a specified alpha value.

    Args:
        base: The base PIL Image.
        overlay: PIL Image or list of PIL Images to overlay on top of the base image.
        alpha: The alpha value for blending (0.0 to 1.0).

    Returns:
        A new PIL Image with all overlays applied.
    """
    # Ensure base image is in RGBA mode
    base = base.convert("RGBA")

    if not isinstance(overlay, list):
        overlay = [overlay]

    for ov in overlay:
        # Ensure overlay image is in RGBA mode
        overlayed_image = ov.convert("RGBA")

        # Resize overlay image to match base image size if necessary
        if base.size != overlayed_image.size:
            overlayed_image = overlayed_image.resize(base.size)

        # Create a mask for non-black pixels in the overlay
        mask = overlayed_image.split()[3]  # Get the alpha channel
        mask = mask.point(lambda p: p > 0 and 255)  # Create binary mask

        # Blend only the non-black pixels of the overlay with the base image
        blended = Image.composite(
            image1=Image.blend(base, overlayed_image, alpha),
            image2=base,
            mask=mask,
        )
        base = blended

    # Ensure the final result is in RGB mode for compatibility
    return base.convert("RGB")


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


def visualize_mask(mask: Image.Image | torch.Tensor, color: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Visualize a mask by applying a color to it.

    Args:
        mask (Image.Image | torch.Tensor): The input mask. Can be a PIL Image or a PyTorch tensor.
        color (tuple[int, int, int], optional): The color to apply to the mask. Defaults to (255, 255, 255) (white).

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
    if field in {"gt_mask"}:
        image = visualize_mask(value)
    if field in {"pred_mask"}:
        image = visualize_mask(value, color=(255, 0, 0))
    if field == "anomaly_map":
        image = visualize_anomaly_map(value, normalize=normalize, colormap=colormap)

    return image


def visualize_image_item(
    item: ImageItem,
    fields: list[str],
    *,  # Mark the following arguments as keyword-only
    field_size: tuple[int, int] = (256, 256),
    overlay_fields: list[tuple[str, list[str]]] | None = None,
    alpha: float = 0.2,
    colormap: bool = True,
    normalize: bool = False,
) -> Image.Image | None:
    """Visualize specified fields of an ImageItem with optional field overlays."""
    images = []
    field_images = {}

    # Collect all fields that need to be visualized
    all_fields = set(fields)
    if overlay_fields:
        for base, overlays in overlay_fields:
            all_fields.add(base)
            all_fields.update(overlays)

    # Visualize all required fields
    for field in all_fields:
        # NOTE: Once pre-processing is implemented, remove this if-else block
        if field == "image":
            image = Image.open(item.image_path)
        else:
            value = getattr(item, field)
            image = visualize_field(field, value, colormap=colormap, normalize=normalize)

        if image:
            image = image.resize(field_size)
            field_images[field] = image

            if field in fields:
                title = convert_to_title_case(field)
                image = add_text_to_image(image, title)
                images.append(image)

    # Process overlay fields
    if overlay_fields:
        for base, overlays in overlay_fields:
            if base in field_images:
                base_image = field_images[base].copy()
                valid_overlays = [overlay for overlay in overlays if overlay in field_images]
                for overlay in valid_overlays:
                    base_image = overlay_images(base_image, field_images[overlay], alpha)

                if valid_overlays:
                    title = (
                        f"{convert_to_title_case(base)} + {'+'.join(convert_to_title_case(o) for o in valid_overlays)}"
                    )
                    base_image = add_text_to_image(base_image, title)
                    images.append(base_image)

    if not images:
        logger.warning("No valid fields to visualize.")
        return None

    return create_image_grid(images, nrow=len(images))
