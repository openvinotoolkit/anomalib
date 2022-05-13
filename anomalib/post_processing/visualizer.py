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
from typing import Dict, List, Optional

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

        self.images: List[Dict] = []

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

    def generate(self):
        """Generate the image."""
        num_cols = len(self.images)
        figure_size = (num_cols * 3, 3)
        self.figure, self.axis = plt.subplots(1, num_cols, figsize=figure_size)
        self.figure.subplots_adjust(right=0.9)

        axes = self.axis if len(self.images) > 1 else [self.axis]
        for axis, image_dict in zip(axes, self.images):
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
