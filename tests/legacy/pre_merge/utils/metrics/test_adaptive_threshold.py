"""Tests for the adaptive threshold metric."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import random

import pytest
import torch
from tests.legacy.helpers.config import get_test_configurable_parameters

from anomalib.data import get_datamodule
from anomalib.engine import Engine
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks
from anomalib.utils.metrics import F1AdaptiveThreshold


@pytest.mark.parametrize(
    ["labels", "preds", "target_threshold"],
    [
        (torch.Tensor([0, 0, 0, 1, 1]), torch.Tensor([2.3, 1.6, 2.6, 7.9, 3.3]), 3.3),  # standard case
        (torch.Tensor([1, 0, 0, 0]), torch.Tensor([4, 3, 2, 1]), 4),  # 100% recall for all thresholds
    ],
)
def test_adaptive_threshold(labels, preds, target_threshold):
    """Test if the adaptive threshold computation returns the desired value."""

    adaptive_threshold = F1AdaptiveThreshold(default_value=0.5)
    adaptive_threshold.update(preds, labels)
    threshold_value = adaptive_threshold.compute()

    assert threshold_value == target_threshold


def test_manual_threshold():
    """
    Test if the manual threshold gets used in the F1 score computation when
    adaptive thresholding is disabled and no normalization is used.
    """
    config = get_test_configurable_parameters(config_path="src/anomalib/models/padim/config.yaml")

    config.data.init_args.num_workers = 0
    config.normalization.normalization_method = "none"
    config.trainer.fast_dev_run = True
    config.metrics.image = ["F1Score"]
    config.metrics.pixel = ["F1Score"]

    image_threshold = 0.12345  # random.random()  # nosec: B311
    pixel_threshold = 0.189761  # random.random()  # nosec: B311
    config.metrics.threshold = [
        {"class_path": "ManualThreshold", "init_args": {"default_value": image_threshold}},
        {"class_path": "ManualThreshold", "init_args": {"default_value": pixel_threshold}},
    ]

    model = get_model(config)
    datamodule = get_datamodule(config)
    callbacks = get_callbacks(config)

    engine = Engine(
        **config.trainer,
        callbacks=callbacks,
        normalization=config.normalization.normalization_method,
        threshold=config.metrics.threshold,
        image_metrics=config.metrics.get("image", None),
        pixel_metrics=config.metrics.get("pixel", None),
        visualization=config.visualization,
    )
    engine.fit(model=model, datamodule=datamodule)
    assert engine.trainer.model.image_metrics.F1Score.threshold == image_threshold
    assert engine.trainer.model.pixel_metrics.F1Score.threshold == pixel_threshold
