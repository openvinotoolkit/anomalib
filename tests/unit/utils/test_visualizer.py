"""Tests for the Visualizer class."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torch.utils.data import DataLoader

from anomalib import TaskType
from anomalib.data import MVTec, PredictDataset
from anomalib.engine import Engine
from anomalib.models import get_model
from anomalib.utils.visualization.image import _ImageGrid


def test_visualize_fully_defected_masks() -> None:
    """Test if a fully defected anomaly mask results in a completely white image."""
    # create visualizer and add fully defected mask
    visualizer = _ImageGrid()
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
    """Test visualization callback for test and predict with different task types."""

    @staticmethod
    @pytest.mark.parametrize("task", [TaskType.CLASSIFICATION, TaskType.SEGMENTATION, TaskType.DETECTION])
    def test_model_visualizer_mode(
        ckpt_path: Callable[[str], Path],
        project_path: Path,
        dataset_path: Path,
        task: TaskType,
    ) -> None:
        """Test combination of model/visualizer/mode on only 1 epoch as a sanity check before merge."""
        _ckpt_path: Path = ckpt_path("Padim")
        model = get_model("padim")
        engine = Engine(
            default_root_dir=project_path,
            fast_dev_run=True,
            devices=1,
            task=task,
        )
        datamodule = MVTec(root=dataset_path / "mvtec", category="dummy", task=task)
        engine.test(model=model, datamodule=datamodule, ckpt_path=str(_ckpt_path))

        dataset = PredictDataset(path=dataset_path / "mvtec" / "dummy" / "test")
        datamodule = DataLoader(dataset)
        engine.predict(model=model, dataloaders=datamodule, ckpt_path=str(_ckpt_path))
