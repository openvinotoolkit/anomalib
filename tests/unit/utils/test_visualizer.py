"""Tests for the Visualizer class."""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torch.utils.data import DataLoader

from anomalib.data import ImageBatch, MVTecAD, PredictDataset
from anomalib.engine import Engine
from anomalib.models import Padim
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
    def test_model_visualizer_mode(
        ckpt_path: Callable[[str], Path],
        project_path: Path,
        dataset_path: Path,
    ) -> None:
        """Test combination of model/visualizer/mode on only 1 epoch as a sanity check before merge."""
        _ckpt_path: Path = ckpt_path("Padim")
        model = Padim(evaluator=False)
        engine = Engine(
            default_root_dir=project_path,
            fast_dev_run=True,
            devices=1,
        )
        datamodule = MVTecAD(root=dataset_path / "mvtecad", category="dummy")
        engine.test(model=model, datamodule=datamodule, ckpt_path=str(_ckpt_path))

        dataset = PredictDataset(path=dataset_path / "mvtecad" / "dummy" / "test")
        datamodule = DataLoader(dataset, collate_fn=ImageBatch.collate)
        engine.predict(model=model, dataloaders=datamodule, ckpt_path=str(_ckpt_path))
