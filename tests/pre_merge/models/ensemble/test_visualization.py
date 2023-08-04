"""Test for ensemble visualizer"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from anomalib.models.ensemble.post_processing import EnsembleVisualization
from tests.helpers.dataset import get_dataset_path

mock_result = {
    "image_path": [Path(get_dataset_path()) / "bottle/test/broken_large/000.png"],
    "image": torch.rand((1, 3, 100, 100)),
    "mask": torch.zeros((1, 100, 100)),
    "anomaly_maps": torch.ones((1, 100, 100)),
    "label": torch.Tensor([0]),
    "pred_scores": torch.Tensor([0.5]),
    "pred_labels": torch.Tensor([0]),
    "pred_masks": torch.zeros((1, 100, 100)),
    "pred_boxes": [torch.rand(1, 4)],
    "box_labels": [torch.tensor([0.5])],
}


@pytest.mark.parametrize("task", ["segmentation", "classification", "detection"])
def test_save_image(task, get_config):
    config = get_config
    with TemporaryDirectory() as temp_dir:
        config.project.path = temp_dir
        config.dataset.task = task
        visualization = EnsembleVisualization(config)
        visualization.process(mock_result)

        assert (Path(temp_dir) / "images/broken_large/000.png").exists()
