"""Anomaly Visualization."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import cv2
import numpy as np
from matplotlib.figure import Figure


class Visualizer:
    """Class that handles the logic of composing the visualizations."""

    @staticmethod
    def show(title: str, image: np.ndarray | Figure, delay: int = 0) -> None:
        """Show an image on the screen.

        Args:
            title (str): Title that will be given to the window showing the image.
            image (np.ndarray | Figure): Image that will be shown in the window.
            delay (int): Delay in milliseconds to wait for keystroke. 0 for infinite.
        """
        if isinstance(image, Figure):
            image = Visualizer.figure_to_array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow(title, image)
        cv2.waitKey(delay)
        cv2.destroyAllWindows()

    @staticmethod
    def save(file_path: Path, image: np.ndarray | Figure) -> None:
        """Save an image to the file system.

        Args:
            file_path (Path): Path to which the image will be saved.
            image (np.ndarray | Figure): Image that will be saved to the file system.
        """
        if isinstance(image, Figure):
            image = Visualizer.figure_to_array(image)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(file_path), image)

    @staticmethod
    def figure_to_array(fig: Figure) -> np.ndarray:
        """Convert a matplotlib figure to a numpy array.

        Args:
            fig (Figure): Matplotlib figure.

        Returns:
            np.ndarray: Numpy array containing the image.
        """
        fig.canvas.draw()
        # convert figure to np.ndarray for saving via visualizer
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
