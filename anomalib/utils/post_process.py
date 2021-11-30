"""Post Process This module contains utils function to apply post-processing to the output predictions."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.


import cv2
import numpy as np
from skimage import morphology


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
    return anomaly_map


def superimpose_anomaly_map(
    anomaly_map: np.ndarray, image: np.ndarray, alpha: float = 0.4, gamma: int = 0
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

    Returns:
        np.ndarray: Image with anomaly map superimposed on top of it.
    """

    anomaly_map = anomaly_map_to_color_map(anomaly_map.squeeze())
    superimposed_map = cv2.addWeighted(anomaly_map, alpha, image, (1 - alpha), gamma)
    superimposed_map = cv2.cvtColor(superimposed_map, cv2.COLOR_BGR2RGB)
    return superimposed_map


def compute_mask(anomaly_map: np.ndarray, threshold: float, kernel_size: int = 4) -> np.ndarray:
    """Compute anomaly mask via thresholding the predicted anomaly map.

    Args:
        anomaly_map: Anomaly map predicted via the model
        threshold: Value to threshold anomaly scores into 0-1 range.
        kernel_size: Value to apply morphological operations to the predicted mask
        anomaly_map: np.ndarray:
        threshold: float:
        kernel_size: int:  (Default value = 4)

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
