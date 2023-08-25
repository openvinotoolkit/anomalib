"""Post Process This module contains utils function to apply post-processing to the output predictions."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum

import cv2
import numpy as np
from skimage import morphology


class ThresholdMethod(str, Enum):
    """Threshold method to apply post-processing to the output predictions."""

    ADAPTIVE = "adaptive"
    MANUAL = "manual"


def add_label(
    image: np.ndarray,
    label_name: str,
    color: tuple[int, int, int],
    confidence: float | None = None,
    font_scale: float = 5e-3,
    thickness_scale=1e-3,
) -> np.ndarray:
    """Adds a label to an image.

    Args:
        image (np.ndarray): Input image.
        label_name (str): Name of the label that will be displayed on the image.
        color (tuple[int, int, int]): RGB values for background color of label.
        confidence (float | None): confidence score of the label.
        font_scale (float): scale of the font size relative to image size. Increase for bigger font.
        thickness_scale (float): scale of the font thickness. Increase for thicker font.

    Returns:
        np.ndarray: Image with label.
    """
    image = image.copy()
    img_height, img_width, _ = image.shape

    font = cv2.FONT_HERSHEY_PLAIN
    text = label_name if confidence is None else f"{label_name} ({confidence*100:.0f}%)"

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
    """Adds the normal label to the image."""
    return add_label(image, "normal", (225, 252, 134), confidence)


def add_anomalous_label(image: np.ndarray, confidence: float | None = None) -> np.ndarray:
    """Adds the anomalous label to the image."""
    return add_label(image, "anomalous", (255, 100, 100), confidence)


def _validate_color(color: tuple[int, int, int]) -> None:
    """TODO move to where? or use another existing one?"""

    if not isinstance(color, Sequence):
        raise ValueError(f"Expected a sequence, but got {type(color).__name__}")

    if len(color) != 3:
        raise ValueError(f"Expected a sequence of length 3, but got a sequence of length {len(color)}")

    channel: int
    for channelidx, channel in enumerate(color):
        if not isinstance(channel, int):
            raise ValueError(f"Expected an integer, but got {type(channel).__name__} on channel {channelidx}")

        if not 0 <= channel <= 255:
            raise ValueError(f"Expected a value in [0, 255], but got {channel} on channel {channelidx}")


def _validate_saturation_colors(saturation_colors: tuple[tuple[int, int, int], tuple[int, int, int]]) -> None:
    """TODO move to where? or use another existing one?"""

    if not isinstance(saturation_colors, Sequence):
        raise ValueError(f"Expected a sequence, but got {type(saturation_colors).__name__}")

    if len(saturation_colors) != 2:
        raise ValueError(f"Expected a sequence of length 2, but got a sequence of length {len(saturation_colors)}")

    color: tuple[int, int, int]
    for coloridx, color in enumerate(saturation_colors):
        try:
            _validate_color(color)

        except ValueError as ex:
            raise ValueError(f"Got an invalid color on position {coloridx}") from ex


def _validate_image(image: np.ndarray) -> None:
    """TODO move to where? or use another existing one?"""

    if not isinstance(image, np.ndarray):
        raise ValueError(f"Expected a numpy array, but got {type(image).__name__}")

    if image.ndim != 3:
        raise ValueError(f"Expected a 3D array, but got {image.ndim}D array")

    if image.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel image, but got {image.shape[2]} channels")

    if image.dtype != np.uint8:
        raise ValueError(f"Expected a uint8 image, but got {image.dtype} image")


def _validate_anomaly_map(anomaly_map: np.ndarray) -> None:
    """TODO move to where? or use another existing one?"""

    if not isinstance(anomaly_map, np.ndarray):
        raise ValueError(f"Expected a numpy array, but got {type(anomaly_map).__name__}")

    if anomaly_map.ndim != 2:
        raise ValueError(f"Expected a 2D array, but got {anomaly_map.ndim}D array")

    if not np.issubdtype(anomaly_map.dtype, np.floating):
        raise ValueError(f"Expected a floating point array, but got {anomaly_map.dtype} array")


def _validate_normalization_bounds(bounds: tuple[float, float]) -> None:
    """TODO move to where? or use another existing one?"""

    if not isinstance(bounds, Sequence):
        raise ValueError(f"Expected a sequence, but got {type(bounds).__name__}")

    if len(bounds) != 2:
        raise ValueError(f"Expected a sequence of length 2, but got a sequence of length {len(bounds)}")

    lower_bound: float
    upper_bound: float
    lower_bound, upper_bound = bounds  # type:ignore

    if not isinstance(lower_bound, float) or not isinstance(upper_bound, float):
        raise ValueError(
            "Expected lower/upper bounds to be floats, "
            f"but got {type(lower_bound).__name__}/{type(upper_bound).__name__}"
        )

    if upper_bound <= lower_bound:
        raise ValueError(
            "Expected upper bound to be greater than lower bound, "
            f"but got (uppper_bound={upper_bound}) <= (lower_bound={lower_bound})"
        )


def _validate_and_convert_normalize(
    normalize: bool | tuple[float, float], anomaly_map: np.ndarray
) -> tuple[float, float] | None:
    """TODO move to where? or use another existing one?"""

    if normalize:
        return (anomaly_map.min(), anomaly_map.max())

    if ((anomaly_map < 0) | (anomaly_map > 1)).any():
        raise Exception("When `normalize` is False, anomaly map values are expected to be in [0, 1]")

    return None


def anomaly_map_to_color_map(
    anomaly_map: np.ndarray,
    normalize: bool | tuple[float, float] = True,
    saturation_colors: tuple[tuple[int, int, int], tuple[int, int, int]] = ((0, 0, 0), (255, 255, 255)),  # black/white
) -> np.ndarray:
    """Compute anomaly color heatmap.

    Gets an array of anomaly scores maps and returns a color map according to a normalized scale.

    Args:
        anomaly_map (np.ndarray): Final anomaly map computed by the distance metric.
        normalize (bool | tuple[float, float], optional): it can work on three modes:
            1) (default) bool and True: normalize the anomaly map with its own min/max
            2) bool and False: values in `anomaly_map` are expected to be \\in [0, 1] (it will be asserted)
            3) tuple[float, float]: values are expected to be anomaly scores lower and upper bounds; in this case
                                    these values are used to rescale the scores \in [0, 1] and whatever is
                                    below/above the lower/upper bound is coloured in black/white.
        saturation_colors (optional): pair of colors in uint8 RGB format to use to saturate the heatmap below and above
                                      (default is black/white as described above)
    Returns:
        np.ndarray: [description]
    """

    _validate_anomaly_map(anomaly_map)

    if isinstance(normalize, bool):
        bounds = _validate_and_convert_normalize(normalize, anomaly_map)
    else:
        _validate_normalization_bounds(normalize)
        bounds = normalize

    if bounds is None:
        anomaly_map_scaled = anomaly_map

    else:
        _validate_saturation_colors(saturation_colors)

        lower_bound, upper_bound = bounds  # normalize the anomaly map
        anomaly_map_scaled = ((anomaly_map - lower_bound) / (upper_bound - lower_bound)).clip(0, 1)

        saturation_color_below, saturation_color_above = saturation_colors

    color_map = cv2.applyColorMap((anomaly_map_scaled * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

    if bounds is None:
        return color_map

    # saturations below and above the score bounds
    over_mask = anomaly_map > upper_bound
    under_mask = anomaly_map < lower_bound

    color_map[over_mask, :] = saturation_color_above  # white (default)
    color_map[under_mask, :] = saturation_color_below  # black (default)

    return color_map


def superimpose_anomaly_map(
    anomaly_map: np.ndarray,
    image: np.ndarray,
    alpha: float = 0.4,
    gamma: int = 0,
    normalize: bool | tuple[float, float] = False,
    saturation_colors: tuple[tuple[int, int, int], tuple[int, int, int]] = ((0, 0, 0), (255, 255, 255)),  # black/white
    ignore_low_scores: bool = True,
) -> np.ndarray:
    """Superimpose anomaly map on top of in the input image.
    Args:
        anomaly_map (np.ndarray): Anomaly map
        image (np.ndarray): Input image
        alpha (float, optional): Weight to overlay anomaly map
            on the input image. Defaults to 0.4.
        gamma (int, optional): Value to add to the blended image
            to smooth the processing. Defaults to 0. Overall,
            the formula to compute the blended image is
            I' = (alpha*I1 + (1-alpha)*I2) + gamma
        normalize (bool | tuple[float, float], optional): it can work on three modes:
            1) (default) bool and True: normalize the anomaly map with its own min/max
            2) bool and False: values in `anomaly_map` are expected to be \\in [0, 1] (it will be asserted)
            3) tuple[float, float]: values are expected to be anomaly scores lower and upper bounds; in this case
                                    these values are used to rescale the scores \in [0, 1] and whatever is
                                    below/above the lower/upper bound is coloured in black/white.
        saturation_colors (optional): pair of colors in uint8 RGB format to use to saturate the heatmap below and above
                                      (default is black/white as described above)
        ignore_low_scores (bool, optional): only used when `normalize` is in the case 3 above;
                                            if true, any score below the lower bound is made transparent

    Returns:
        np.ndarray: Image with anomaly map superimposed on top of it.
    """

    _validate_image(image)
    _validate_anomaly_map(anomaly_map)

    if anomaly_map.shape != image.shape[:2]:
        raise Exception()

    # there was a `anomaly_map.squeeze()` before --> do something about it? now it is validated to be squeezed
    color_map = anomaly_map_to_color_map(anomaly_map, normalize=normalize, saturation_colors=saturation_colors)
    superimposed_map = cv2.addWeighted(color_map, alpha, image, (1 - alpha), gamma)

    if isinstance(normalize, bool) or not ignore_low_scores:
        return superimposed_map

    # when bounds are given, make anything under the lower bound transparent
    lower_bound, _ = normalize
    under_mask = anomaly_map < lower_bound

    # repaint the image over the pixels with score below the lower bound (as if the heatmap was transparent)
    superimposed_map[under_mask] = image[under_mask]

    return superimposed_map


def compute_mask(anomaly_map: np.ndarray, threshold: float, kernel_size: int = 4) -> np.ndarray:
    """Compute anomaly mask via thresholding the predicted anomaly map.

    Args:
        anomaly_map (np.ndarray): Anomaly map predicted via the model
        threshold (float): Value to threshold anomaly scores into 0-1 range.
        kernel_size (int): Value to apply morphological operations to the predicted mask. Defaults to 4.

    Returns:
        Predicted anomaly mask
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

    Args:
        image (np.ndarray): Source image.
        boxes (np.nparray): 2D array of shape (N, 4) where each row contains the xyxy coordinates of a bounding box.
        color (tuple[int, int, int]): Color of the drawn boxes in RGB format.

    Returns:
        np.ndarray: Image showing the bounding boxes drawn on top of the source image.
    """
    for box in boxes:
        x_1, y_1, x_2, y_2 = box.astype(int)
        image = cv2.rectangle(image, (x_1, y_1), (x_2, y_2), color=color, thickness=2)
    return image
