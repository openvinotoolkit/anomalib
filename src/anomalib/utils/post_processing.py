"""Post Process This module contains utils function to apply post-processing to the output predictions."""

# Copyright (C) 2022-2024 Intel Corporation
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
    """Add a label to an image.

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
    """Add the normal label to the image."""
    return add_label(image, "normal", (225, 252, 134), confidence)


def add_anomalous_label(image: np.ndarray, confidence: float | None = None) -> np.ndarray:
    """Add the anomalous label to the image."""
    return add_label(image, "anomalous", (255, 100, 100), confidence)


def anomaly_map_to_color_map(anomaly_map: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Compute anomaly color heatmap.

    Args:
        anomaly_map (np.ndarray): Final anomaly map computed by the distance metric.
        normalize (bool, optional): Bool to normalize the anomaly map prior to applying
            the color map. Defaults to True.

    Returns:
        np.ndarray: [description]
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
        normalize: whether or not the anomaly maps should
            be normalized to image min-max at image level


    Returns:
        np.ndarray: Image with anomaly map superimposed on top of it.
    """
    anomaly_map = anomaly_map_to_color_map(anomaly_map.squeeze(), normalize=normalize)
    return cv2.addWeighted(anomaly_map, alpha, image, (1 - alpha), gamma)


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
