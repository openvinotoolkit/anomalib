"""Tests for the adaptive threshold metric."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
import random

import pytest
import torch
from pytorch_lightning import Trainer

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks
from anomalib.utils.metrics import AdaptiveThreshold


@pytest.mark.parametrize(
    ["labels", "preds", "target_threshold"],
    [
        (torch.Tensor([0, 0, 0, 1, 1]), torch.Tensor([2.3, 1.6, 2.6, 7.9, 3.3]), 3.3),  # standard case
        (torch.Tensor([1, 0, 0, 0]), torch.Tensor([4, 3, 2, 1]), 4),  # 100% recall for all thresholds
    ],
)
def test_adaptive_threshold(labels, preds, target_threshold):
    """Test if the adaptive threshold computation returns the desired value."""

    adaptive_threshold = AdaptiveThreshold(default_value=0.5)
    adaptive_threshold.update(preds, labels)
    threshold_value = adaptive_threshold.compute()

    assert threshold_value == target_threshold


def test_non_adaptive_threshold():
    """
    Test if the non-adaptive threshold gets used in the F1 score computation when
    adaptive thresholding is disabled and no normalization is used.
    """
    config = get_configurable_parameters(model_config_path="anomalib/models/padim/config.yaml")

    config.model.normalization_method = "none"
    config.model.threshold.adaptive = False
    config.trainer.fast_dev_run = True

    image_threshold = random.random()
    pixel_threshold = random.random()
    config.model.threshold.image_default = image_threshold
    config.model.threshold.pixel_default = pixel_threshold

    model = get_model(config)
    datamodule = get_datamodule(config)
    callbacks = get_callbacks(config)

    trainer = Trainer(**config.trainer, callbacks=callbacks)
    trainer.fit(model=model, datamodule=datamodule)
    assert trainer.model.image_metrics.F1.threshold == image_threshold
    assert trainer.model.pixel_metrics.F1.threshold == pixel_threshold
