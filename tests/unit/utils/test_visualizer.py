# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Visualizer class."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from anomalib.utils.visualization.image import _ImageGrid


def test_visualize_fully_defected_masks() -> None:
    """Test if a fully defected anomaly mask results in a completely white image."""
    visualizer = _ImageGrid()
    mask = np.ones((256, 256)) * 255
    visualizer.add_image(image=mask, color_map="gray", title="fully defected mask")
    visualizer.generate()

    canvas = FigureCanvas(visualizer.figure)
    canvas.draw()
    plotted_img = visualizer.axis.images[0].make_image(canvas.renderer)

    assert np.all(plotted_img[0][..., 0] == 255)
