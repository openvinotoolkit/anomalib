"""Dummy model that is used to test teh visualizer callback."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import torch
from torch import nn

from anomalib import LearningType
from anomalib.models.components import AnomalyModule


class _DummyModel(nn.Module): ...


class DummyModule(AnomalyModule):
    """A dummy model which calls visualizer callback on fake images and masks.

    TODO(ashwinvaidya17): Remove this when the DummyModels have been refactored.
    """

    def __init__(self, dataset_path: Path) -> None:
        """Initializes the dummy model."""
        super().__init__()
        self.model = _DummyModel()
        self.task = "segmentation"
        self.mode = "full"
        self.dataset_path = dataset_path

    def validation_step(self, batch: dict[str, str | torch.Tensor], *args, **kwargs) -> dict:
        """Only used to avoid NotImplementedError."""
        del batch
        return self.test_step(*args, **kwargs)

    def test_step(self, *_, **__) -> dict:
        """Only used to trigger on_test_epoch_end."""
        self.log(name="loss", value=0.0, prog_bar=True)
        return {
            "image_path": [Path(self.dataset_path / "mvtec" / "dummy" / "train" / "good" / "000.png")],
            "image": torch.rand((1, 3, 100, 100)).to(self.device),
            "mask": torch.zeros((1, 100, 100)).to(self.device),
            "anomaly_maps": torch.ones((1, 100, 100)).to(self.device),
            "label": torch.Tensor([0]).to(self.device),
            "pred_labels": torch.Tensor([0]).to(self.device),
            "pred_masks": torch.zeros((1, 100, 100)).to(self.device),
        }

    def configure_optimizers(self) -> None:
        """Optimization is not required."""

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        """Does not require anything specific."""
        return {}

    @property
    def learning_type(self) -> LearningType:
        """Returns the learning type."""
        return LearningType.ZERO_SHOT
