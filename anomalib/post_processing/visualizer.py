"""Anomaly Visualization."""

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

import math
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """Anomaly Visualization.

    The visualizer object is responsible for collating all the images passed to it into a single image. This can then
    either be logged by accessing the `figure` attribute or can be saved directly by calling `save()` method.

    Example:
        >>> visualizer = Visualizer()
        >>> visualizer.add_image(image=image, title="Image")
        >>> visualizer.close()
    """

    def __init__(self):

        self.images = []

        self.figure: matplotlib.figure.Figure
        self.axis: np.ndarray

    def add_image(self, image: np.ndarray, title: str, color_map: Optional[str] = None):
        """Add image to figure.

        Args:
          image (np.ndarray): Image which should be added to the figure.
          title (str): Image title shown on the plot.
          color_map (Optional[str]): Name of matplotlib color map used to map scalar data to colours. Defaults to None.
        """
        image_data = dict(image=image, title=title, color_map=color_map)
        self.images.append(image_data)

    def add_text(self, image: np.ndarray, text: str, font: int = cv2.FONT_HERSHEY_PLAIN):
        """Puts text on an image.

        Args:
            image (np.ndarray): Input image.
            text (str): Text to add.
            font (Optional[int]): cv2 font type. Defaults to 0.

        Returns:
            np.ndarray: Image with text.
        """
        image = image.copy()
        font_size = image.shape[1] // 256 + 1  # Text scale is calculated based on the reference size of 256

        for i, line in enumerate(text.split("\n")):
            (text_w, text_h), baseline = cv2.getTextSize(line.strip(), font, font_size, thickness=1)
            offset = i * text_h
            cv2.rectangle(image, (0, offset + baseline // 2), (0 + text_w, 0 + text_h + offset), (255, 255, 255), -1)
            cv2.putText(image, line.strip(), (0, (baseline // 2 + text_h) + offset), font, font_size, (0, 0, 255))
        return image

    @staticmethod
    def add_label(
        image: np.ndarray,
        label_name: str,
        color: Tuple[int, int, int],
        confidence: Optional[float] = None,
        font_scale: float = 5e-3,
        thickness_scale=1e-3,
    ):
        """Adds a label to an image.

        Args:
            image (np.ndarray): Input image.
            label_name (str): Name of the label that will be displayed on the image.
            color (Tuple[int, int, int]): RGB values for background color of label.
            confidence (Optional[float]): confidence score of the label.
            font_scale (float): scale of the font size relative to image size. Increase for bigger font.
            thickness_scale (float): scale of the font thickness. Increase for thicker font.

        Returns:
            np.ndarray: Image with label.
        """
        image = image.copy()
        img_height, img_width, _ = image.shape

        font = cv2.FONT_HERSHEY_PLAIN
        text = label_name if confidence is None else f"{label_name}: {confidence:.2}"

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

    def add_normal_label(self, image: np.ndarray, confidence: Optional[float] = None):
        """Adds the normal label to the image."""
        return self.add_label(image, "normal", (225, 252, 134), confidence)

    def add_anomalous_label(self, image: np.ndarray, confidence: Optional[float] = None):
        """Adds the anomalous label to the image."""
        return self.add_label(image, "anomalous", (255, 100, 100), confidence)

    def generate(self):
        """Generate the image."""
        num_cols = len(self.images)
        figure_size = (num_cols * 3, 3)
        self.figure, self.axis = plt.subplots(1, num_cols, figsize=figure_size)
        self.figure.subplots_adjust(right=0.9)

        for axis, image_dict in zip(self.axis, self.images):
            axis.axes.xaxis.set_visible(False)
            axis.axes.yaxis.set_visible(False)
            axis.imshow(image_dict["image"], image_dict["color_map"], vmin=0, vmax=255)
            axis.title.set_text(image_dict["title"])

    def show(self):
        """Show image on a matplotlib figure."""
        self.generate()
        self.figure.show()

    def save(self, filename: Path):
        """Save image.

        Args:
          filename (Path): Filename to save image
        """
        self.generate()
        filename.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(filename, dpi=100)

    def close(self):
        """Close figure."""
        plt.close(self.figure)
