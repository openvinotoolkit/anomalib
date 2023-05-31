"""Base normalizer."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from abc import ABC
from typing import Type

from pytorch_lightning.utilities.types import STEP_OUTPUT
from torchmetrics import Metric

from anomalib import trainer


class BaseNormalizer(ABC):
    """Base class for normalizers.

    Args:
        trainer (trainer.AnomalibTrainer): Trainer object.
    """

    def __init__(self, trainer: "trainer.AnomalibTrainer"):
        self.metric_class: Type[Metric]
        self.trainer = trainer
        self._metric: Metric | None = None

    @property
    def metric(self) -> Metric:
        """Assigns metrics if not available."""
        # get metric from trainer
        if self._metric is None:
            self._metric = self.metric_class()
        # Every time metric is called, it is moved to the cpu
        return self._metric.cpu()

    @metric.setter
    def metric(self, metric: Metric) -> None:
        """Sets the metric.

        Args:
            metric (Metric): Metric to set.
        """
        self._metric = metric

    def update(self, outputs: STEP_OUTPUT):
        raise NotImplementedError("update not implemented in the child class.")

    def normalize(self, outputs: STEP_OUTPUT):
        raise NotImplementedError("normalize not implemented in the child class.")
