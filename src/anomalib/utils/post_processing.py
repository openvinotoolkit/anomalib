"""Post-processing utilities for anomaly detection predictions.

This module provides utilities for post-processing anomaly detection predictions.
The key components include:

    - Label addition to images with confidence scores
    - Morphological operations on prediction masks
    - Normalization and thresholding of anomaly maps

Example:
    >>> import numpy as np
    >>> from anomalib.utils.post_processing import add_label
    >>> # Add label to image
    >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
    >>> labeled_image = add_label(
    ...     image=image,
    ...     label_name="Anomalous",
    ...     color=(255, 0, 0),
    ...     confidence=0.95
    ... )

The module ensures consistent post-processing by:
    - Providing standardized label formatting
    - Supporting both classification and segmentation outputs
    - Handling proper scaling of visual elements
    - Offering configurable processing parameters

Note:
    All functions preserve the input data types and handle proper normalization
    of values where needed.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math

import cv2
import numpy as np
from skimage import morphology


def add_label(
    image: np.ndarray,
    label_name: str,
    color: tuple[int, int, int],
    confidence: float | None = None,
    font_scale: float = 5e-3,
    thickness_scale: float = 1e-3,
) -> np.ndarray:
    """Add a text label with optional confidence score to an image.

    This function adds a text label to the top-left corner of an image. The label has a
    colored background patch and can optionally include a confidence percentage.

    Args:
        image (np.ndarray): Input image to add the label to. Must be a 3-channel RGB or
            BGR image.
        label_name (str): Text label to display on the image (e.g. "normal",
            "anomalous").
        color (tuple[int, int, int]): RGB color values for the label background as a
            tuple of 3 integers in range [0,255].
        confidence (float | None, optional): Confidence score between 0 and 1 to display
            as percentage. If ``None``, only the label name is shown. Defaults to
            ``None``.
        font_scale (float, optional): Scale factor for font size relative to image
            dimensions. Larger values produce bigger text. Defaults to ``5e-3``.
        thickness_scale (float, optional): Scale factor for font thickness relative to
            image dimensions. Larger values produce thicker text. Defaults to ``1e-3``.

    Returns:
        np.ndarray: Copy of input image with label added to top-left corner.

    Example:
        Add a normal label with 95% confidence:

        >>> import numpy as np
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> labeled_image = add_label(
        ...     image=image,
        ...     label_name="normal",
        ...     color=(0, 255, 0),
        ...     confidence=0.95
        ... )

        Add an anomalous label without confidence:

        >>> labeled_image = add_label(
        ...     image=image,
        ...     label_name="anomalous",
        ...     color=(255, 0, 0)
        ... )

    Note:
        - The function creates a copy of the input image to avoid modifying it
        - Font size and thickness scale automatically with image dimensions
        - Label is always placed in the top-left corner
        - Uses OpenCV's FONT_HERSHEY_PLAIN font family
    """
    image = image.copy()
    img_height, img_width, _ = image.shape

    font = cv2.FONT_HERSHEY_PLAIN
    text = label_name if confidence is None else f"{label_name} ({confidence * 100:.0f}%)"

    # get font sizing
    font_scale = min(img_width, img_height) * font_scale
    thickness = math.ceil(min(img_width, img_height) * thickness_scale)
    (width, height), baseline = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)

    # create label
    label_patch = np.zeros((height + baseline, width + baseline, 3), dtype=np.uint8)
    label_patch[:, :] = color
    cv2.putText(
        label_patch,
        text,
        (0, baseline // 2 + height),
        font,
        fontScale=font_scale,
        thickness=thickness,
        color=0,
        lineType=cv2.LINE_AA,
    )

    # add label to image
    image[: baseline + height, : baseline + width] = label_patch
    return image


def add_normal_label(image: np.ndarray, confidence: float | None = None) -> np.ndarray:
    """Add a 'normal' label to the image.

    This function adds a 'normal' label to the top-left corner of the image using a
    light green color. The label can optionally include a confidence score.

    Args:
        image (np.ndarray): Input image to add the label to. Should be a 3-channel
            RGB or BGR image.
        confidence (float | None, optional): Confidence score between 0 and 1 to
            display with the label. If ``None``, only the label is shown.
            Defaults to ``None``.

    Returns:
        np.ndarray: Copy of input image with 'normal' label added.

    Examples:
        Add normal label without confidence:

        >>> labeled_image = add_normal_label(image)

        Add normal label with 95% confidence:

        >>> labeled_image = add_normal_label(image, confidence=0.95)

    Note:
        - Creates a copy of the input image
        - Uses a light green color (RGB: 225, 252, 134)
        - Label is placed in top-left corner
        - Font size scales with image dimensions
    """
    return add_label(image, "normal", (225, 252, 134), confidence)


def add_anomalous_label(image: np.ndarray, confidence: float | None = None) -> np.ndarray:
    """Add an 'anomalous' label to the image.

    This function adds an 'anomalous' label to the top-left corner of the image using a
    light red color. The label can optionally include a confidence score.

    Args:
        image (np.ndarray): Input image to add the label to. Should be a 3-channel
            RGB or BGR image.
        confidence (float | None, optional): Confidence score between 0 and 1 to
            display with the label. If ``None``, only the label is shown.
            Defaults to ``None``.

    Returns:
        np.ndarray: Copy of input image with 'anomalous' label added.

    Examples:
        Add anomalous label without confidence:

        >>> labeled_image = add_anomalous_label(image)

        Add anomalous label with 95% confidence:

        >>> labeled_image = add_anomalous_label(image, confidence=0.95)

    Note:
        - Creates a copy of the input image
        - Uses a light red color (RGB: 255, 100, 100)
        - Label is placed in top-left corner
        - Font size scales with image dimensions
    """
    return add_label(image, "anomalous", (255, 100, 100), confidence)


def anomaly_map_to_color_map(anomaly_map: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Convert an anomaly map to a color heatmap visualization.

    This function converts a grayscale anomaly map into a color heatmap using the JET
    colormap. The anomaly map can optionally be normalized before coloring.

    Args:
        anomaly_map (np.ndarray): Grayscale anomaly map computed by the model's
            distance metric. Should be a 2D array of float values.
        normalize (bool, optional): Whether to normalize the anomaly map to [0,1] range
            before applying the colormap. If ``True``, the map is normalized using
            min-max scaling. Defaults to ``True``.

    Returns:
        np.ndarray: RGB color heatmap visualization of the anomaly map. Values are in
            range [0,255] and type uint8.

    Examples:
        Convert anomaly map without normalization:

        >>> heatmap = anomaly_map_to_color_map(anomaly_map, normalize=False)
        >>> heatmap.shape
        (224, 224, 3)
        >>> heatmap.dtype
        dtype('uint8')

        Convert with normalization (default):

        >>> heatmap = anomaly_map_to_color_map(anomaly_map)
        >>> heatmap.min(), heatmap.max()
        (0, 255)

    Note:
        - Input map is converted to uint8 by scaling to [0,255] range
        - Uses OpenCV's JET colormap for visualization
        - Output is converted from BGR to RGB color format
        - Shape of output matches input with added channel dimension
    """
    if normalize:
        anomaly_map = (anomaly_map - anomaly_map.min()) / np.ptp(anomaly_map)
    anomaly_map = anomaly_map * 255
    anomaly_map = anomaly_map.astype(np.uint8)

    anomaly_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    return cv2.cvtColor(anomaly_map, cv2.COLOR_BGR2RGB)


def superimpose_anomaly_map(
    anomaly_map: np.ndarray,
    image: np.ndarray,
    alpha: float = 0.4,
    gamma: int = 0,
    normalize: bool = False,
) -> np.ndarray:
    """Superimpose an anomaly heatmap on top of an input image.

    This function overlays a colored anomaly map visualization on an input image using
    alpha blending. The anomaly map can optionally be normalized before blending.

    Args:
        anomaly_map (np.ndarray): Grayscale anomaly map computed by the model's
            distance metric. Should be a 2D array of float values.
        image (np.ndarray): Input image to overlay the anomaly map on. Will be
            resized to match anomaly map dimensions.
        alpha (float, optional): Blending weight for the anomaly map overlay.
            Should be in range [0,1] where 0 shows only the input image and 1
            shows only the anomaly map. Defaults to ``0.4``.
        gamma (int, optional): Value added to the blended result for smoothing.
            The blending formula is:
            ``output = (alpha * anomaly_map + (1-alpha) * image) + gamma``
            Defaults to ``0``.
        normalize (bool, optional): Whether to normalize the anomaly map to [0,1]
            range before coloring. If ``True``, uses min-max scaling at the image
            level. Defaults to ``False``.

    Returns:
        np.ndarray: RGB image with the colored anomaly map overlay. Values are in
            range [0,255] and type uint8.

    Examples:
        Basic overlay without normalization:

        >>> result = superimpose_anomaly_map(anomaly_map, image, alpha=0.4)
        >>> result.shape
        (224, 224, 3)
        >>> result.dtype
        dtype('uint8')

        Overlay with normalization and custom blending:

        >>> result = superimpose_anomaly_map(
        ...     anomaly_map,
        ...     image,
        ...     alpha=0.7,
        ...     gamma=10,
        ...     normalize=True
        ... )

    Note:
        - Input image is resized to match anomaly map dimensions
        - Anomaly map is converted to a color heatmap using JET colormap
        - Output maintains RGB color format
        - Shape of output matches the anomaly map dimensions
    """
    anomaly_map = anomaly_map_to_color_map(anomaly_map.squeeze(), normalize=normalize)
    height, width = anomaly_map.shape[:2]
    image = cv2.resize(image, (width, height))
    return cv2.addWeighted(anomaly_map, alpha, image, (1 - alpha), gamma)


def compute_mask(anomaly_map: np.ndarray, threshold: float, kernel_size: int = 4) -> np.ndarray:
    """Compute binary anomaly mask by thresholding and post-processing anomaly map.

    This function converts a continuous-valued anomaly map into a binary mask by:
        - Thresholding the anomaly scores
        - Applying morphological operations to reduce noise
        - Scaling to 8-bit range [0, 255]

    Args:
        anomaly_map (np.ndarray): Anomaly map containing predicted anomaly scores.
            Should be a 2D array of float values.
        threshold (float): Threshold value to binarize anomaly scores. Values above
            this threshold are considered anomalous (1) and below as normal (0).
        kernel_size (int, optional): Size of the morphological structuring element
            used for noise removal. Higher values result in smoother masks.
            Defaults to ``4``.

    Returns:
        np.ndarray: Binary anomaly mask where anomalous regions are marked with
            255 and normal regions with 0. Output is uint8 type.

    Examples:
        Basic thresholding with default kernel size:

        >>> anomaly_scores = np.random.rand(100, 100)
        >>> mask = compute_mask(anomaly_scores, threshold=0.5)
        >>> mask.shape
        (100, 100)
        >>> mask.dtype
        dtype('uint8')
        >>> np.unique(mask)
        array([  0, 255], dtype=uint8)

        Custom kernel size for stronger smoothing:

        >>> mask = compute_mask(anomaly_scores, threshold=0.5, kernel_size=8)

    Note:
        - Input anomaly map is squeezed to remove singleton dimensions
        - Morphological opening is used to remove small noise artifacts
        - Output is scaled to [0, 255] range for visualization
    """
    anomaly_map = anomaly_map.squeeze()
    mask: np.ndarray = np.zeros_like(anomaly_map).astype(np.uint8)
    mask[anomaly_map > threshold] = 1

    kernel = morphology.disk(kernel_size)
    mask = morphology.opening(mask, kernel)

    mask *= 255

    return mask


def draw_boxes(image: np.ndarray, boxes: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """Draw bounding boxes on an image.

    This function draws rectangular bounding boxes on an input image using OpenCV. Each box
    is drawn with the specified color and a fixed thickness of 2 pixels.

    Args:
        image (np.ndarray): Source image on which to draw the boxes. Should be a valid
            OpenCV-compatible image array.
        boxes (np.ndarray): 2D array of shape ``(N, 4)`` where each row contains the
            ``(x1, y1, x2, y2)`` coordinates of a bounding box in pixel units. The
            coordinates specify the top-left and bottom-right corners.
        color (tuple[int, int, int]): Color of the drawn boxes in RGB format, specified
            as a tuple of 3 integers in the range ``[0, 255]``.

    Returns:
        np.ndarray: Modified image with bounding boxes drawn on top. Has the same
            dimensions and type as the input image.

    Examples:
        Draw a single red box:

        >>> import numpy as np
        >>> image = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> boxes = np.array([[10, 10, 50, 50]])  # Single box
        >>> result = draw_boxes(image, boxes, color=(255, 0, 0))
        >>> result.shape
        (100, 100, 3)

        Draw multiple boxes in green:

        >>> boxes = np.array([
        ...     [20, 20, 40, 40],
        ...     [60, 60, 80, 80]
        ... ])  # Two boxes
        >>> result = draw_boxes(image, boxes, color=(0, 255, 0))

    Note:
        - Input coordinates are converted to integers before drawing
        - Boxes are drawn with a fixed thickness of 2 pixels
        - The function modifies the input image in-place
        - OpenCV uses BGR color format internally but the function expects RGB
    """
    for box in boxes:
        x_1, y_1, x_2, y_2 = box.astype(int)
        image = cv2.rectangle(image, (x_1, y_1), (x_2, y_2), color=color, thickness=2)
    return image
