"""Tests for the adaptive threshold metric."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import random

import pytest
import torch
from pytorch_lightning import Trainer

from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks
from anomalib.utils.metrics import AnomalyScoreThreshold
from tests.helpers.config import get_test_configurable_parameters


@pytest.mark.parametrize(
    ["labels", "preds", "target_threshold"],
    [
        (torch.Tensor([0, 0, 0, 1, 1]), torch.Tensor([2.3, 1.6, 2.6, 7.9, 3.3]), 3.3),  # standard case
        (torch.Tensor([1, 0, 0, 0]), torch.Tensor([4, 3, 2, 1]), 4),  # 100% recall for all thresholds
    ],
)
def test_adaptive_threshold(labels, preds, target_threshold):
    """Test if the adaptive threshold computation returns the desired value."""

    adaptive_threshold = AnomalyScoreThreshold(default_value=0.5)
    adaptive_threshold.update(preds, labels)
    threshold_value = adaptive_threshold.compute()

    assert threshold_value == target_threshold


def test_manual_threshold():
    """
    Test if the manual threshold gets used in the F1 score computation when
    adaptive thresholding is disabled and no normalization is used.
    """
    config = get_test_configurable_parameters(config_path="anomalib/models/padim/config.yaml")

    config.model.normalization_method = "none"
    config.metrics.threshold.method = "manual"
    config.trainer.fast_dev_run = True
    config.metrics.image = ["F1Score"]
    config.metrics.pixel = ["F1Score"]

    image_threshold = random.random()
    pixel_threshold = random.random()
    config.metrics.threshold.manual_image = image_threshold
    config.metrics.threshold.manual_pixel = pixel_threshold

    model = get_model(config)
    datamodule = get_datamodule(config)
    callbacks = get_callbacks(config)

    trainer = Trainer(**config.trainer, callbacks=callbacks)
    trainer.fit(model=model, datamodule=datamodule)
    assert trainer.model.image_metrics.F1Score.threshold == image_threshold
    assert trainer.model.pixel_metrics.F1Score.threshold == pixel_threshold
