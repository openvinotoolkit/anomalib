"""Tests for the AnomalibMetric base class."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torchmetrics import Metric

from anomalib.data import ImageBatch
from anomalib.metrics import AnomalibMetric, create_anomalib_metric


class DummyMetric(Metric):
    """Dummy metric that does nothing."""

    def update(self, *args, **kwargs) -> None:
        """Dummy update method."""

    def compute(self) -> None:
        """Dummy compute method."""


class TestMetricCreation:
    """Test the creation of Anomalib metrics."""

    @staticmethod
    def test_create_anomalib_metric_function() -> None:
        """Test if defining a metric using the function works."""
        metric_cls = create_anomalib_metric(DummyMetric)
        assert issubclass(metric_cls, AnomalibMetric)
        assert issubclass(metric_cls, Metric)

    @staticmethod
    def test_create_anomalib_metric_subclass() -> None:
        """Test if defining a metric using a subclass works."""

        class AnomalibDummyMetric(AnomalibMetric, DummyMetric):
            pass

        assert issubclass(AnomalibDummyMetric, AnomalibMetric)
        assert issubclass(AnomalibDummyMetric, Metric)


class TestStrictMode:
    """Test the strict mode of Anomalib metrics."""

    @staticmethod
    def test_raises_error_on_missing_fields() -> None:
        """Test that an error is raised when required fields are missing in strict mode."""
        metric_cls = create_anomalib_metric(DummyMetric)

        metric = metric_cls(fields=["non_existent_field"])
        batch = ImageBatch(image=torch.rand(4, 3, 10, 10))  # batch without field
        with pytest.raises(ValueError, match="instance is missing required field"):
            metric.update(batch)
        assert metric._update_count == 0  # noqa: SLF001
        assert metric.update_called is False

    @staticmethod
    def test_raises_error_when_field_is_none() -> None:
        """Test that an error is raised when a required field is None in strict mode."""
        metric_cls = create_anomalib_metric(DummyMetric)

        metric = metric_cls(fields=["pred_score"])
        batch = ImageBatch(image=torch.rand(4, 3, 10, 10), pred_score=None)  # batch where field is None
        with pytest.raises(ValueError, match="instance does not have a value for field with name"):
            metric.update(batch)
        assert metric._update_count == 0  # noqa: SLF001
        assert metric.update_called is False

    @staticmethod
    def test_no_error_on_missing_fields() -> None:
        """Test that no error is raised when required fields are missing in non-strict mode."""
        metric_cls = create_anomalib_metric(DummyMetric)

        metric = metric_cls(fields=["pred_score"], strict=False)
        batch = ImageBatch(image=torch.rand(4, 3, 10, 10))  # batch without pred_score field
        metric.update(batch)
        assert metric.compute() is None
        assert metric._update_count == 0  # noqa: SLF001
        assert metric.update_called is False

    @staticmethod
    def test_no_error_when_field_is_none() -> None:
        """Test that no error is raised when a required field is None in non-strict mode."""
        metric_cls = create_anomalib_metric(DummyMetric)

        metric = metric_cls(fields=["pred_score"], strict=False)
        batch = ImageBatch(image=torch.rand(4, 3, 10, 10), pred_score=None)
        metric.update(batch)
        assert metric.compute() is None
        assert metric._update_count == 0  # noqa: SLF001
        assert metric.update_called is False
