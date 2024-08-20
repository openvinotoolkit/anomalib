"""Tests for the adaptive threshold metric."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.metrics import F1AdaptiveThreshold
from anomalib.models import Padim
from anomalib.utils.normalization import NormalizationMethod


@pytest.mark.parametrize(
    ("labels", "preds", "target_threshold"),
    [
        (torch.Tensor([0, 0, 0, 1, 1]), torch.Tensor([2.3, 1.6, 2.6, 7.9, 3.3]), 3.3),  # standard case
        (torch.Tensor([1, 0, 0, 0]), torch.Tensor([4, 3, 2, 1]), 4),  # 100% recall for all thresholds
    ],
)
def test_adaptive_threshold(labels: torch.Tensor, preds: torch.Tensor, target_threshold: int | float) -> None:
    """Test if the adaptive threshold computation returns the desired value."""
    adaptive_threshold = F1AdaptiveThreshold(default_value=0.5)
    adaptive_threshold.update(preds, labels)
    threshold_value = adaptive_threshold.compute()

    assert threshold_value == target_threshold


def test_manual_threshold() -> None:
    """Test manual threshold.

    Test if the manual threshold gets used in the F1 score computation when
    adaptive thresholding is disabled and no normalization is used.
    """
    image_threshold = 0.12345  # random.random()  # nosec: B311
    pixel_threshold = 0.189761  # random.random()  # nosec: B311
    threshold = [
        {"class_path": "ManualThreshold", "init_args": {"default_value": image_threshold}},
        {"class_path": "ManualThreshold", "init_args": {"default_value": pixel_threshold}},
    ]

    model = Padim()
    datamodule = MVTec()

    engine = Engine(
        normalization=NormalizationMethod.NONE,
        threshold=threshold,
        image_metrics="F1Score",
        pixel_metrics="F1Score",
        fast_dev_run=True,
        accelerator="gpu",
        devices=1,
    )
    engine.fit(model=model, datamodule=datamodule)
    assert engine.trainer.lightning_module.image_metrics.F1Score.threshold == image_threshold
    assert engine.trainer.lightning_module.pixel_metrics.F1Score.threshold == pixel_threshold
