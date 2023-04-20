"""Base normalizer."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import Type

from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import Metric

import anomalib.trainer as trainer
from anomalib.models import AnomalyModule


class BaseNormalizer(ABC):
    """Base class for normalizers.

    Args:
        trainer (trainer.AnomalibTrainer): Trainer object.
    """

    def __init__(self, trainer: "trainer.AnomalibTrainer"):
        self.metric_class: Type[Metric]
        self.trainer = trainer

    @property
    def anomaly_module(self) -> AnomalyModule:
        """Returns the anomaly module."""
        if not hasattr(self.trainer, "lightning_module"):
            raise ValueError("Lightning module is not available yet in trainer.")
        return self.trainer.lightning_module

    @property
    def metric(self) -> Metric:
        """Assigns metrics if not available."""
        # get metric from anomaly_module
        if not hasattr(self.trainer.lightning_module, "normalization_metrics"):
            self.trainer.lightning_module.normalization_metrics = self.metric_class()
        # Every time metric is called, it is moved to the cpu
        return self.trainer.lightning_module.normalization_metrics.cpu()

    def update(self, outputs: STEP_OUTPUT):
        raise NotImplementedError("update not implemented in the child class.")

    def normalize(self, outputs: STEP_OUTPUT):
        raise NotImplementedError("normalize not implemented in the child class.")
