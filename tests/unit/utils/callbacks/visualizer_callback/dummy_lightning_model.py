"""Dummy model that is used to test teh visualizer callback."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import torch
from torch import nn

from anomalib import LearningType
from anomalib.data import ImageBatch, InferenceBatch
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor


class _DummyModel(nn.Module): ...


class DummyPostProcessor(PostProcessor):
    """Dummy post-processor for testing."""

    @staticmethod
    def forward(batch: InferenceBatch) -> InferenceBatch:
        """Dummy forward method."""
        return batch


class DummyModule(AnomalibModule):
    """A dummy model which calls visualizer callback on fake images and masks.

    TODO(ashwinvaidya17): Remove this when the DummyModels have been refactored.
    """

    def __init__(self, dataset_path: Path, **kwargs) -> None:
        """Initializes the dummy model."""
        super().__init__(**kwargs)
        self.model = _DummyModel()
        self.task = "segmentation"
        self.mode = "full"
        self.dataset_path = dataset_path

    def validation_step(self, batch: ImageBatch, *args, **kwargs) -> ImageBatch:
        """Only used to avoid NotImplementedError."""
        del batch
        return self.test_step(*args, **kwargs)

    def test_step(self, *_, **__) -> ImageBatch:
        """Only used to trigger on_test_epoch_end."""
        self.log(name="loss", value=0.0, prog_bar=True)
        return ImageBatch(
            image_path=[Path(self.dataset_path / "mvtecad" / "dummy" / "train" / "good" / "000.png")],
            image=torch.rand((1, 3, 100, 100)).to(self.device),
            gt_mask=torch.zeros((1, 100, 100)).to(self.device),
            anomaly_map=torch.ones((1, 100, 100)).to(self.device),
            pred_score=torch.Tensor([1.0]),
            gt_label=torch.Tensor([0]).int().to(self.device),
            pred_label=torch.Tensor([0]).int().to(self.device),
            pred_mask=torch.zeros((1, 100, 100)).to(self.device),
        )

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

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        """Returns a dummy post-processor."""
        return DummyPostProcessor()
