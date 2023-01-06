"""Tests for the Visualizer class."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tempfile

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from anomalib.data import TaskType
from anomalib.post_processing.visualizer import ImageGrid
from tests.helpers.dataset import TestDataset
from tests.helpers.model import setup_model_train


def test_visualize_fully_defected_masks():
    """Test if a fully defected anomaly mask results in a completely white image."""

    # create visualizer and add fully defected mask
    visualizer = ImageGrid()
    mask = np.ones((256, 256)) * 255
    visualizer.add_image(image=mask, color_map="gray", title="fully defected mask")
    visualizer.generate()

    # retrieve plotted image
    canvas = FigureCanvas(visualizer.figure)
    canvas.draw()
    plotted_img = visualizer.axis.images[0].make_image(canvas.renderer)

    # assert that the plotted image is completely white
    assert np.all(plotted_img[0][..., 0] == 255)


class TestVisualizer:
    @pytest.mark.parametrize(
        ["model_name", "nncf"],
        [
            ("padim", False),
            ("ganomaly", False),
        ],
    )
    @pytest.mark.parametrize("task", (TaskType.CLASSIFICATION, TaskType.SEGMENTATION, TaskType.DETECTION))
    @pytest.mark.parametrize("mode", ("full", "simple"))
    @TestDataset(num_train=20, num_test=10)
    def test_model_visualizer_mode(self, model_name, nncf, task, mode, category="shapes", path=""):
        """Test combination of model/visualizer/mode on only 1 epoch as a sanity check before merge."""
        with tempfile.TemporaryDirectory() as project_path:
            # Train test
            datamodule, model, trainer = setup_model_train(
                model_name,
                dataset_path=path,
                project_path=project_path,
                nncf=nncf,
                category=category,
                fast_run=True,
                dataset_task=task,
                visualizer_mode=mode,
            )[1:]
            trainer.test(model=model, datamodule=datamodule)
