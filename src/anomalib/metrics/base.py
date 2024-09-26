"""Base classes for metrics in Anomalib."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Sequence
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import BinaryF1Score
from anomalib.data import Batch


class AnomalibMetric:
    """Base class for metrics in Anomalib.
    
    This class is designed to be a base class for all metrics in Anomalib. It
    adds the ability to update the metric with a Batch object, which is a
    container for the predictions and targets of a model. When instantiating
    a new metric, the user must provide a list of fields that the metric will
    use to update its state. The metric will then use the Batch object to
    extract the values of these fields and pass them to the
    `update` method of the metric.
    """

    def __init__(self, fields: Sequence[str], prefix: str = "", **kwargs) -> None:
        self.fields = fields
        self.name = prefix + self.__class__.__name__
        super().__init__(**kwargs)

    def __init_subclass__(cls, **kwargs) -> None:
        del kwargs
        assert issubclass(cls, (Metric, MetricCollection)), "AnomalibMetric must be a subclass of torchmetrics.Metric or torchmetrics.MetricCollection"

    def update(self, batch: Batch, *args, **kwargs):
        values = [getattr(batch, key) for key in self.fields]
        super().update(*values, *args, **kwargs)

class MetricWrapper(AnomalibMetric, MetricCollection):
    pass

class F1Score(AnomalibMetric, BinaryF1Score):
    pass
