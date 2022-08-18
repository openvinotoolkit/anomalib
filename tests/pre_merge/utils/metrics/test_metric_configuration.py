"""Tests for checking if the right metrics are loaded."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from anomalib.models import AnomalyModule
from anomalib.utils.callbacks.metrics_configuration import MetricsConfigurationCallback


def test_thresholding():
    """
    Tests the available thresholding methods.
    """

    # Test adaptive
    module = AnomalyModule()
    threshold = "adaptive"
    callback = MetricsConfigurationCallback(threshold, None, None)
    callback.setup(Trainer(), module)
    assert module.threshold == "adaptive"
    assert module.image_threshold == 0.0
    assert module.pixel_threshold == 0.0

    module = AnomalyModule()
    threshold = DictConfig({"adaptive": {"default_value": 0.5}})
    callback = MetricsConfigurationCallback(threshold, None, None)
    callback.setup(Trainer(), module)
    assert module.threshold == "adaptive"
    assert module.image_threshold == 0.5
    assert module.pixel_threshold == 0.5

    # Test manual
    module = AnomalyModule()
    threshold = "manual"
    callback = MetricsConfigurationCallback(threshold, None, None)
    # Should raise error as default thresholds are not provided
    with pytest.raises(AssertionError):
        callback.setup(Trainer(), module)

    module = AnomalyModule()
    threshold = threshold = DictConfig({"manual": {"image_threshold": 0.5, "pixel_threshold": 1.5}})
    callback = MetricsConfigurationCallback(threshold, None, None)
    callback.setup(Trainer(), module)
    assert module.image_threshold == 0.5
    assert module.pixel_threshold == 1.5

    module = AnomalyModule()
    threshold = threshold = DictConfig({"manual": {"image_threshold": 0.5}})
    callback = MetricsConfigurationCallback(threshold, None, None)
    callback.setup(Trainer(), module)
    assert module.image_threshold == 0.5
    assert module.pixel_threshold == 0.5

    # Test maximum
    module = AnomalyModule()
    threshold = "maximum"
    callback = MetricsConfigurationCallback(threshold, None, None)
    callback.setup(Trainer(), module)
    test_tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    module.image_threshold.update(test_tensor, torch.zeros(1))
    assert module.threshold == "maximum"
    assert module.image_threshold == 10.0
    assert module.pixel_threshold == 10.0
