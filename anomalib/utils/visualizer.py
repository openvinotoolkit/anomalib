"""
Anomaly Visualization
"""

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """
    Anomaly Visualization
    """

    def __init__(self, num_rows: int, num_cols: int, figure_size: Tuple[int, int]):
        self.figure_index: int = 0

        self.figure, self.axis = plt.subplots(num_rows, num_cols, figsize=figure_size)
        self.figure.subplots_adjust(right=0.9)

        for axis in self.axis:
            axis.axes.xaxis.set_visible(False)
            axis.axes.yaxis.set_visible(False)

    def add_image(self, image: np.ndarray, title: str, color_map: Optional[str] = None, index: Optional[int] = None):
        """
        Add image to figure

        Args:
          image: np.ndarray:
          title: str: Image title shown on the plot.
          color_map: Optional[str]:  (Default value = None)
          index: Optional[int]:  (Default value = None)

        """
        if index is None:
            index = self.figure_index
            self.figure_index += 1

        self.axis[index].imshow(image, color_map)
        self.axis[index].title.set_text(title)

    def show(self):
        """
        Show image on a matplotlib figure.
        """
        self.figure.show()

    def save(self, filename: Path):
        """
        Save image.

        Args:
          filename: Path: Filename to save image
        """
        filename.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(filename, dpi=100)

    def close(self):
        """
        Close figure.
        """
        plt.close(self.figure)
