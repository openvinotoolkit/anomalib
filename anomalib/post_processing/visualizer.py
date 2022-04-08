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

from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """Anomaly Visualization.

    The visualizer object is responsible for collating all the images passed to it into a single image. This can then
    either be logged by accessing the `figure` attribute or can be saved directly by calling `save()` method.

    Example:
        >>> visualizer = Visualizer(num_rows=1, num_cols=5, figure_size=(12, 3))
        >>> visualizer.add_image(image=image, title="Image")
        >>> visualizer.close()

    Args:
        num_rows (int): Number of rows of images in the figure.
        num_cols (int): Number of columns/images in each row.
        figure_size (Tuple[int, int]): Size of output figure
    """

    def __init__(self, num_rows: int, num_cols: int, figure_size: Tuple[int, int]):
        self.figure_index: int = 0

        self.figure, self.axis = plt.subplots(num_rows, num_cols, figsize=figure_size)
        self.figure.subplots_adjust(right=0.9)

        for axis in self.axis:
            axis.axes.xaxis.set_visible(False)
            axis.axes.yaxis.set_visible(False)

    def add_image(self, image: np.ndarray, title: str, color_map: Optional[str] = None, index: Optional[int] = None):
        """Add image to figure.

        Args:
          image (np.ndarray): Image which should be added to the figure.
          title (str): Image title shown on the plot.
          color_map (Optional[str]): Name of matplotlib color map used to map scalar data to colours. Defaults to None.
          index (Optional[int]): Figure index. Defaults to None.
        """
        if index is None:
            index = self.figure_index
            self.figure_index += 1

        self.axis[index].imshow(image, color_map, vmin=0, vmax=255)
        self.axis[index].title.set_text(title)

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

    def show(self):
        """Show image on a matplotlib figure."""
        self.figure.show()

    def save(self, filename: Path):
        """Save image.

        Args:
          filename (Path): Filename to save image
        """
        filename.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(filename, dpi=100)

    def close(self):
        """Close figure."""
        plt.close(self.figure)
