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
from anomalib.utils.visualization.image import ImageResult, ImageVisualizer, VisualizationMode, _ImageGrid


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


def test_model_visualizer_visual_prompting() -> None:
    """Test visualizer image on TaskType.VISUAL_PROMPTING."""
    anomaly_map = np.zeros((100, 100), dtype=np.float64)
    anomaly_map[10:20, 10:20] = 1.0
    gt_mask = np.zeros((100, 100))
    gt_mask[15:25, 15:25] = 1.0
    rng = np.random.default_rng()
    image = rng.integers(0, 255, size=(100, 100, 3), dtype=np.uint8)

    image_result = ImageResult(
        image=image,
        pred_score=0.9,
        pred_label="abnormal",
        text_descr=(
            "Some very long text to see how it is formatted in the image"
            " Some very long text to see how it is formatted in the image"
        ),
        anomaly_map=anomaly_map,
        gt_mask=gt_mask,
        pred_mask=anomaly_map,
    )

    image_visualizer = ImageVisualizer(
        mode=VisualizationMode.FULL,
        task=TaskType.VISUAL_PROMPTING,
    )
    result = image_visualizer.visualize_image(image_result)

    assert result.shape == (500, 500, 3)


class TestVisualizer:
    """Test visualization callback for test and predict with different task types."""

    @pytest.mark.parametrize("task", [TaskType.CLASSIFICATION, TaskType.SEGMENTATION, TaskType.DETECTION])
    def test_model_visualizer_mode(
        self,
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
