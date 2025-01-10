"""Image visualization functions using PIL and torchvision.

This module provides functions for visualizing images and anomaly detection results using
PIL and torchvision. The key components include:

    - Functions for adding text overlays to images
    - Tools for applying colormaps to anomaly maps
    - Image overlay and blending utilities
    - Mask and anomaly map visualization

Example:
    >>> from PIL import Image
    >>> from anomalib.visualization.image.functional import add_text_to_image
    >>> # Create image and add text
    >>> image = Image.new('RGB', (100, 100))
    >>> result = add_text_to_image(image, text="Anomaly")

The module ensures consistent visualization by:
    - Providing standardized text rendering
    - Supporting various color formats and fonts
    - Handling different image formats
    - Maintaining aspect ratios

Note:
    All visualization functions preserve the input image format and dimensions
    unless explicitly specified otherwise.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import inspect
import logging
import sys
from collections.abc import Callable
from typing import Any, Literal

import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont
from torchvision.transforms.functional import to_pil_image

logger = logging.getLogger(__name__)


def dynamic_font_size(image_size: tuple[int, int], min_size: int = 20, max_size: int = 100, divisor: int = 10) -> int:
    """Calculate a dynamic font size based on image dimensions.

    This function determines an appropriate font size based on the image dimensions while
    staying within specified bounds. The font size is calculated by dividing the smaller
    image dimension by the divisor.

    Args:
        image_size (tuple[int, int]): Tuple of image dimensions ``(width, height)``.
        min_size (int, optional): Minimum allowed font size. Defaults to ``20``.
        max_size (int, optional): Maximum allowed font size. Defaults to ``100``.
        divisor (int, optional): Value to divide the minimum image dimension by.
            Defaults to ``10``.

    Returns:
        int: Calculated font size constrained between ``min_size`` and ``max_size``.

    Examples:
        Calculate font size for a small image:

        >>> dynamic_font_size((200, 100))
        20

        Calculate font size for a large image:

        >>> dynamic_font_size((1000, 800))
        80

        Font size is capped at max_size:

        >>> dynamic_font_size((2000, 2000), max_size=50)
        50

    Note:
        - The function uses the smaller dimension to ensure text fits in both directions
        - The calculated size is clamped between ``min_size`` and ``max_size``
        - Larger ``divisor`` values result in smaller font sizes
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
    """Add text to an image with configurable parameters.

    This function adds text to a PIL Image with customizable font, size, color and
    background options. The text can be positioned anywhere on the image and includes
    an optional background box.

    Args:
        image (Image.Image): The PIL Image to add text to.
        text (str): The text string to add to the image.
        font (str | None, optional): Path to a font file. If ``None`` or loading fails,
            the default system font is used. Defaults to ``None``.
        size (int | None, optional): Font size in pixels. If ``None``, size is
            calculated dynamically based on image dimensions. Defaults to ``None``.
        color (tuple[int, int, int] | str, optional): Text color as RGB tuple or color
            name. Defaults to ``"white"``.
        background (tuple[int, ...] | str | None, optional): Background color for text
            box. Can be RGB/RGBA tuple or color name. If ``None``, no background is
            drawn. Defaults to semi-transparent black ``(0, 0, 0, 128)``.
        position (tuple[int, int], optional): Top-left position of text as ``(x, y)``
            coordinates. Defaults to ``(10, 10)``.
        padding (int, optional): Padding around text in background box in pixels.
            Defaults to ``3``.

    Returns:
        Image.Image: New PIL Image with text added.

    Examples:
        Basic white text:

        >>> from PIL import Image
        >>> img = Image.new('RGB', (200, 100))
        >>> result = add_text_to_image(img, "Hello")

        Custom font and color:

        >>> result = add_text_to_image(
        ...     img,
        ...     "Hello",
        ...     font="arial.ttf",
        ...     color=(255, 0, 0)
        ... )

        Text with custom background:

        >>> result = add_text_to_image(
        ...     img,
        ...     "Hello",
        ...     background=(0, 0, 255, 200),
        ...     position=(50, 50)
        ... )

    Note:
        - The function creates a transparent overlay for the text
        - Font size is calculated dynamically if not specified
        - Falls back to default system font if custom font fails to load
        - Input image is converted to RGBA for compositing
        - Output is converted back to RGB
    """
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
    The colormap is created by interpolating between 9 key colors from dark blue to dark
    red.

    Args:
        image (``Image.Image``): A single-channel PIL Image or an object that can be
            converted to PIL Image. If not already in 'L' mode (8-bit grayscale), it will
            be converted.

    Returns:
        ``Image.Image``: A new PIL Image in RGB mode with the colormap applied.

    Raises:
        TypeError: If the input cannot be converted to a PIL Image.

    Example:
        Create a random grayscale image and apply colormap:

        >>> from PIL import Image
        >>> import numpy as np
        >>> # Create a sample grayscale image
        >>> gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        >>> gray_image = Image.fromarray(gray, mode='L')
        >>> # Apply the jet colormap
        >>> colored_image = apply_colormap(gray_image)
        >>> colored_image.mode
        'RGB'

        Apply to non-PIL input:

        >>> # NumPy array input is automatically converted
        >>> colored_image = apply_colormap(gray)
        >>> isinstance(colored_image, Image.Image)
        True

        Invalid input raises TypeError:

        >>> apply_colormap("not an image")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        TypeError: Input must be a PIL Image object or an object that can be...

    Note:
        - Input is automatically converted to grayscale if not already
        - Uses a custom 'jet' colormap interpolated between 9 key colors
        - Output is always in RGB mode regardless of input mode
        - The colormap interpolation uses bilinear mode for smooth transitions
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

    This function takes a base image and overlays another image on top of it with the
    specified transparency level. Both images are converted to RGBA mode to enable alpha
    compositing. If the overlay image has a different size than the base image, it will
    be resized to match.

    Args:
        base (:class:`PIL.Image.Image`): The base image that will serve as the
            background.
        overlay (:class:`PIL.Image.Image`): The image to overlay on top of the base
            image.
        alpha (float, optional): The alpha/transparency value for blending, between
            0.0 (fully transparent) and 1.0 (fully opaque). Defaults to ``0.5``.

    Returns:
        :class:`PIL.Image.Image`: A new image with the overlay composited on top of
        the base image using the specified alpha value.

    Examples:
        Create a base image with a yellow triangle on a green background:

        >>> from PIL import Image, ImageDraw
        >>> image = Image.new('RGB', (200, 200), color='green')
        >>> draw = ImageDraw.Draw(image)
        >>> draw.polygon([(50, 50), (150, 50), (100, 150)], fill='yellow')

        Create a mask with a white rectangle on black background:

        >>> mask = Image.new('L', (200, 200), color=0)
        >>> draw = ImageDraw.Draw(mask)
        >>> draw.rectangle([75, 75, 125, 125], fill=255)

        Overlay the mask on the image with 30% opacity:

        >>> result = overlay_image(image, mask, alpha=0.3)
        >>> result.show()  # doctest: +SKIP

    Note:
        - Both input images are converted to RGBA mode internally
        - The overlay is automatically resized to match the base image size
        - The function uses PIL's alpha compositing for high-quality blending
        - The output image preserves the RGBA mode of the composite result
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
    """Overlay multiple images on top of a base image with specified transparency.

    This function overlays one or more images on top of a base image with specified
    alpha/transparency values. If an overlay is a mask (L mode), it will be drawn
    as a semi-transparent overlay.

    Args:
        base (:class:`PIL.Image.Image`): The base image to overlay on top of.
        overlays (:class:`PIL.Image.Image` | list[:class:`PIL.Image.Image`]): One
            or more images to overlay on the base image.
        alpha (float | list[float], optional): Alpha/transparency value(s) between
            0.0 (fully transparent) and 1.0 (fully opaque). Can be a single float
            applied to all overlays, or a list of values. Defaults to ``0.5``.

    Returns:
        :class:`PIL.Image.Image`: A new image with all overlays composited on top
        of the base image using the specified alpha values.

    Examples:
        Overlay a single mask:

        >>> from PIL import Image, ImageDraw
        >>> # Create base image with yellow triangle on green background
        >>> image = Image.new('RGB', (200, 200), color='green')
        >>> draw = ImageDraw.Draw(image)
        >>> draw.polygon([(50, 50), (150, 50), (100, 150)], fill='yellow')
        >>> # Create mask with white rectangle
        >>> mask = Image.new('L', (200, 200), color=0)
        >>> draw = ImageDraw.Draw(mask)
        >>> draw.rectangle([75, 75, 125, 125], fill=255)
        >>> # Apply overlay
        >>> result = overlay_images(image, mask)

        Overlay multiple masks with different alphas:

        >>> # Create second mask with white ellipse
        >>> mask2 = Image.new('L', (200, 200), color=0)
        >>> draw = ImageDraw.Draw(mask2)
        >>> draw.ellipse([50, 50, 150, 100], fill=255)
        >>> # Apply overlays with different alpha values
        >>> result = overlay_images(image, [mask, mask2], alpha=[0.3, 0.7])

    Note:
        - All images are converted to RGBA mode internally
        - Overlays are automatically resized to match the base image size
        - Uses PIL's alpha compositing for high-quality blending
        - The output preserves the RGBA mode of the composite result
    """
    if not isinstance(overlays, list):
        overlays = [overlays]

    alphas = [alpha] * len(overlays) if not isinstance(alpha, list) else alpha
    for overlay, overlay_alpha in zip(overlays, alphas, strict=False):
        base = overlay_image(base, overlay, alpha=overlay_alpha)

    return base


def visualize_anomaly_map(
    anomaly_map: Image.Image | torch.Tensor,
    colormap: bool = True,
    normalize: bool = False,
) -> Image.Image:
    """Visualize an anomaly map by applying normalization and/or colormap.

    This function takes an anomaly map and converts it to a visualization by optionally
    normalizing the values and applying a colormap. The input can be either a PIL Image
    or PyTorch tensor.

    Args:
        anomaly_map (:class:`PIL.Image.Image` | :class:`torch.Tensor`): Input anomaly
            map to visualize. If a tensor is provided, it will be converted to a PIL
            Image.
        colormap (bool, optional): Whether to apply a colormap to the anomaly map.
            When ``True``, converts the image to a colored heatmap visualization.
            When ``False``, converts to RGB grayscale. Defaults to ``True``.
        normalize (bool, optional): Whether to normalize the anomaly map values to
            [0, 255] range before visualization. When ``True``, linearly scales the
            values using min-max normalization. Defaults to ``False``.

    Returns:
        :class:`PIL.Image.Image`: Visualized anomaly map as a PIL Image in RGB mode.
            If ``colormap=True``, returns a heatmap visualization. Otherwise returns
            a grayscale RGB image.

    Examples:
        Visualize a PIL Image anomaly map:

        >>> from PIL import Image
        >>> import numpy as np
        >>> # Create sample anomaly map
        >>> data = np.random.rand(100, 100).astype(np.float32)
        >>> anomaly_map = Image.fromarray(data, mode='F')
        >>> # Visualize with normalization and colormap
        >>> vis = visualize_anomaly_map(anomaly_map, normalize=True, colormap=True)
        >>> vis.mode
        'RGB'

        Visualize a PyTorch tensor anomaly map:

        >>> import torch
        >>> # Create random tensor
        >>> tensor_map = torch.rand(100, 100)
        >>> # Visualize without normalization
        >>> vis = visualize_anomaly_map(tensor_map, normalize=False, colormap=True)
        >>> isinstance(vis, Image.Image)
        True

    Note:
        - Input tensors are automatically converted to PIL Images
        - The function always returns an RGB mode image
        - When ``normalize=True``, uses min-max normalization to [0, 255] range
        - The colormap used is the default from :func:`apply_colormap`
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

    This function takes a binary mask and visualizes it in different styles based on the
    specified mode.

    Args:
        mask (:class:`PIL.Image.Image` | :class:`torch.Tensor`): Input mask to visualize.
            Can be a PIL Image or PyTorch tensor. If tensor, should be 2D with values in
            [0, 1] or [0, 255].
        mode (Literal["contour", "fill", "binary", "L", "1"]): Visualization mode:

            - ``"contour"``: Draw contours around masked regions
            - ``"fill"``: Fill masked regions with semi-transparent color
            - ``"binary"``: Return original binary mask
            - ``"L"``: Return original grayscale mask
            - ``"1"``: Return original binary mask

        alpha (float, optional): Alpha value for blending in ``"fill"`` mode.
            Should be between 0.0 and 1.0. Defaults to ``0.5``.
        color (tuple[int, int, int], optional): RGB color to apply to mask.
            Each value should be 0-255. Defaults to ``(255, 0, 0)`` (red).
        background_color (tuple[int, int, int, int], optional): RGBA background color.
            Each value should be 0-255. Defaults to ``(0, 0, 0, 0)`` (transparent).

    Returns:
        :class:`PIL.Image.Image`: Visualized mask as a PIL Image. The output mode
        depends on the visualization mode:

        - ``"contour"`` and ``"fill"``: Returns RGBA image
        - ``"binary"``, ``"L"``, ``"1"``: Returns grayscale image

    Raises:
        TypeError: If ``mask`` is not a PIL Image or PyTorch tensor.
        ValueError: If ``mode`` is not one of the allowed values.

    Examples:
        Create a random binary mask:

        >>> import numpy as np
        >>> from PIL import Image
        >>> mask_array = np.random.randint(0, 2, size=(100, 100), dtype=np.uint8) * 255
        >>> mask_image = Image.fromarray(mask_array, mode='L')

        Visualize mask contours in red:

        >>> contour_vis = visualize_mask(
        ...     mask_image,
        ...     mode="contour",
        ...     color=(255, 0, 0)
        ... )
        >>> isinstance(contour_vis, Image.Image)
        True

        Fill mask regions with semi-transparent green:

        >>> fill_vis = visualize_mask(
        ...     mask_image,
        ...     mode="fill",
        ...     color=(0, 255, 0),
        ...     alpha=0.3
        ... )
        >>> isinstance(fill_vis, Image.Image)
        True

        Return original binary mask:

        >>> binary_vis = visualize_mask(mask_image, mode="binary")
        >>> binary_vis.mode
        'L'

    Note:
        - Input tensors are automatically converted to PIL Images
        - Binary masks are expected to have values of 0 and 255 (or 0 and 1 for tensors)
        - The function preserves the original mask when using ``"binary"``, ``"L"`` or
          ``"1"`` modes
        - ``"contour"`` mode uses edge detection to find mask boundaries
        - ``"fill"`` mode creates a semi-transparent overlay using the specified color
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
    """Visualize a ground truth mask.

    This is a convenience wrapper around :func:`visualize_mask` specifically for
    ground truth masks. It provides the same functionality with default parameters
    suitable for ground truth visualization.

    Args:
        mask (Image.Image | torch.Tensor): Input mask to visualize. Can be either a
            PIL Image or PyTorch tensor.
        mode (Literal["contour", "fill", "binary", "L", "1"]): Visualization mode.
            Defaults to ``"binary"``.
            - ``"contour"``: Draw mask boundaries
            - ``"fill"``: Fill mask regions with semi-transparent color
            - ``"binary"``, ``"L"``, ``"1"``: Return original binary mask
        alpha (float): Opacity for the mask visualization in ``"fill"`` mode.
            Range [0, 1]. Defaults to ``0.5``.
        color (tuple[int, int, int]): RGB color for visualizing the mask.
            Defaults to red ``(255, 0, 0)``.
        background_color (tuple[int, int, int, int]): RGBA color for the
            background. Defaults to transparent ``(0, 0, 0, 0)``.

    Returns:
        Image.Image: Visualized mask as a PIL Image.

    Examples:
        >>> import torch
        >>> from PIL import Image
        >>> # Create a sample binary mask
        >>> mask = torch.zeros((100, 100))
        >>> mask[25:75, 25:75] = 1
        >>> # Visualize with default settings (binary mode)
        >>> vis = visualize_gt_mask(mask)
        >>> isinstance(vis, Image.Image)
        True
        >>> # Visualize with contours in blue
        >>> vis = visualize_gt_mask(mask, mode="contour", color=(0, 0, 255))
        >>> isinstance(vis, Image.Image)
        True

    Note:
        See :func:`visualize_mask` for more details on the visualization modes and
        parameters.
    """
    return visualize_mask(mask, mode=mode, alpha=alpha, color=color, background_color=background_color)


def visualize_pred_mask(
    mask: Image.Image | torch.Tensor,
    *,
    mode: Literal["contour", "fill", "binary", "L", "1"] = "binary",
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
    background_color: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Image.Image:
    """Visualize a prediction mask.

    This is a convenience wrapper around :func:`visualize_mask` specifically for
    prediction masks. It provides the same functionality with default parameters
    suitable for prediction visualization.

    Args:
        mask (Image.Image | torch.Tensor): Input mask to visualize. Can be either a
            PIL Image or PyTorch tensor.
        mode (Literal["contour", "fill", "binary", "L", "1"]): Visualization mode.
            Defaults to ``"binary"``.
            - ``"contour"``: Draw mask boundaries
            - ``"fill"``: Fill mask regions with semi-transparent color
            - ``"binary"``, ``"L"``, ``"1"``: Return original binary mask
        color (tuple[int, int, int]): RGB color for visualizing the mask.
            Defaults to red ``(255, 0, 0)``.
        alpha (float): Opacity for the mask visualization in ``"fill"`` mode.
            Range [0, 1]. Defaults to ``0.5``.
        background_color (tuple[int, int, int, int]): RGBA color for the
            background. Defaults to transparent ``(0, 0, 0, 0)``.

    Returns:
        Image.Image: Visualized mask as a PIL Image.

    Examples:
        >>> import torch
        >>> from PIL import Image
        >>> # Create a sample binary mask
        >>> mask = torch.zeros((100, 100))
        >>> mask[25:75, 25:75] = 1
        >>> # Visualize with default settings (binary mode)
        >>> vis = visualize_pred_mask(mask)
        >>> isinstance(vis, Image.Image)
        True
        >>> # Visualize with contours in blue
        >>> vis = visualize_pred_mask(mask, mode="contour", color=(0, 0, 255))
        >>> isinstance(vis, Image.Image)
        True

    Note:
        See :func:`visualize_mask` for more details on the visualization modes and
        parameters.
    """
    return visualize_mask(mask, mode=mode, alpha=alpha, color=color, background_color=background_color)


def create_image_grid(images: list[Image.Image], nrow: int) -> Image.Image:
    """Create a grid of images using PIL.

    This function arranges a list of PIL images into a grid layout with a specified
    number of images per row. All input images must have the same dimensions.

    Args:
        images (list[Image.Image]): List of PIL Images to arrange in a grid. All
            images must have identical dimensions.
        nrow (int): Number of images to display per row in the grid.

    Returns:
        Image.Image: A new PIL Image containing the arranged grid of input images
            with white background.

    Raises:
        ValueError: If ``images`` list is empty.

    Examples:
        Create a 2x2 grid from 4 images:

        >>> from PIL import Image
        >>> import numpy as np
        >>> # Create sample images
        >>> img1 = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        >>> img2 = Image.fromarray(np.ones((64, 64, 3), dtype=np.uint8) * 255)
        >>> images = [img1, img2, img1, img2]
        >>> # Create grid with 2 images per row
        >>> grid = create_image_grid(images, nrow=2)
        >>> isinstance(grid, Image.Image)
        True
        >>> grid.size
        (128, 128)

    Note:
        - All input images must have identical dimensions
        - The grid is filled row by row, left to right, top to bottom
        - If the number of images is not divisible by ``nrow``, the last row may
          be partially filled
        - The output image dimensions will be:
          width = ``nrow`` * image_width
          height = ceil(len(images)/nrow) * image_height
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

    This function retrieves the default keyword arguments for a given visualization
    function by inspecting its signature.

    Args:
        field (str): The name of the visualization field (e.g., ``'mask'``,
            ``'anomaly_map'``).

    Returns:
        dict[str, Any]: A dictionary containing the default keyword arguments for
            the visualization function. Each key is a parameter name and the value
            is its default value.

    Raises:
        ValueError: If the specified ``field`` does not have a corresponding
            visualization function in the current module.

    Examples:
        Get keyword arguments for visualizing a mask:

        >>> # Get keyword arguments for visualizing a mask
        >>> mask_kwargs = get_field_kwargs('mask')
        >>> print(mask_kwargs)  # doctest: +SKIP
        {
            'mode': 'binary',
            'color': (255, 0, 0),
            'alpha': 0.5,
            'background_color': (0, 0, 0, 0)
        }

        Get keyword arguments for visualizing an anomaly map:

        >>> # Get keyword arguments for visualizing an anomaly map
        >>> anomaly_map_kwargs = get_field_kwargs('anomaly_map')
        >>> print(anomaly_map_kwargs)  # doctest: +SKIP
        {
            'colormap': True,
            'normalize': False
        }

        Attempt to get keyword arguments for an invalid field:

        >>> get_field_kwargs('invalid_field')  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        ValueError: 'invalid_field' is not a valid function in the current module.

    Note:
        - The function looks for a visualization function named
          ``visualize_{field}`` in the current module
        - Only parameters with default values are included in the returned dict
        - Variable keyword arguments (``**kwargs``) are noted in the dict with a
          descriptive string
        - Both keyword-only and positional-or-keyword parameters are included
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

    This function retrieves the visualization function corresponding to a specified field
    from the current module. The function name is constructed by prepending
    ``visualize_`` to the field name.

    Args:
        field (str): Name of the visualization field. Common values include:
            - ``"image"``: For basic image visualization
            - ``"mask"``: For segmentation mask visualization
            - ``"anomaly_map"``: For anomaly heatmap visualization

    Returns:
        Callable: Visualization function corresponding to the given field.
            The returned function will accept parameters specific to that
            visualization type.

    Raises:
        AttributeError: If no visualization function exists for the specified
            ``field``. The error message will indicate which function name was
            not found.

    Examples:
        Get visualization function for an anomaly map:

        >>> from PIL import Image
        >>> visualize_func = get_visualize_function('anomaly_map')
        >>> anomaly_map = Image.new('F', (256, 256))
        >>> visualized_map = visualize_func(
        ...     anomaly_map,
        ...     colormap=True,
        ...     normalize=True
        ... )
        >>> isinstance(visualized_map, Image.Image)
        True

        Get visualization function for a mask:

        >>> visualize_func = get_visualize_function('mask')
        >>> mask = Image.new('1', (256, 256))
        >>> visualized_mask = visualize_func(mask, color=(255, 0, 0))
        >>> isinstance(visualized_mask, Image.Image)
        True

        Attempting to get function for invalid field raises error:

        >>> get_visualize_function('invalid_field')  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        AttributeError: module 'anomalib.visualization.image.functional' has no
        attribute 'visualize_invalid_field'

    Note:
        - The function looks for visualization functions in the current module
        - Function names must follow the pattern ``visualize_{field}``
        - Each visualization function may have different parameters
        - All visualization functions return PIL Image objects
    """
    current_module = sys.modules[__name__]
    func_name = f"visualize_{field}"
    return getattr(current_module, func_name)
