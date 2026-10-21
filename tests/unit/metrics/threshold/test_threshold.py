# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test Threshold metric."""

import pytest
from torchmetrics import Metric

from anomalib.metrics.threshold import Threshold


class TestThreshold:
    """Test cases for the Threshold class."""

    @staticmethod
    def test_threshold_abstract_methods() -> None:
        """Test that Threshold class raises NotImplementedError for abstract methods."""
        threshold = Threshold()

        with pytest.raises(NotImplementedError, match=r"Subclass of Threshold must implement the compute method"):
            threshold.compute()

        with pytest.raises(NotImplementedError, match=r"Subclass of Threshold must implement the update method"):
            threshold.update()

    @staticmethod
    def test_threshold_initialization() -> None:
        """Test that Threshold can be initialized without errors."""
        threshold = Threshold()
        assert isinstance(threshold, Metric)


class TestBaseThreshold:
    """Test cases for the BaseThreshold class."""

    @staticmethod
    def test_base_threshold_inheritance() -> None:
        """Test that BaseThreshold inherits from Threshold."""
        base_threshold = Threshold()
        assert isinstance(base_threshold, Threshold)

    @staticmethod
    def test_base_threshold_abstract_methods() -> None:
        """Test that BaseThreshold class raises NotImplementedError for abstract methods."""
        base_threshold = Threshold()

        with pytest.raises(NotImplementedError, match=r"Subclass of Threshold must implement the compute method"):
            base_threshold.compute()

        with pytest.raises(NotImplementedError, match=r"Subclass of Threshold must implement the update method"):
            base_threshold.update()


class TestF1AdaptiveThreshold:
    """Test cases for F1AdaptiveThreshold class."""

    @staticmethod
    def test_normal_samples_only() -> None:
        """Test threshold computation when only normal samples are present."""
        import torch
        from anomalib.metrics.threshold.f1_adaptive_threshold import _F1AdaptiveThreshold

        metric = _F1AdaptiveThreshold()
        preds = torch.tensor([0.1, 0.2, 0.3])
        targets = torch.tensor([0, 0, 0])
        metric.update(preds, targets)
        threshold = metric.compute()
        assert threshold == torch.tensor(0.3)

    @staticmethod
    def test_anomalous_samples_only() -> None:
        """Test threshold computation when only anomalous samples are present."""
        import torch
        from anomalib.metrics.threshold.f1_adaptive_threshold import _F1AdaptiveThreshold

        metric = _F1AdaptiveThreshold()
        preds = torch.tensor([0.7, 0.8, 0.9])
        targets = torch.tensor([1, 1, 1])
        metric.update(preds, targets)
        threshold = metric.compute()
        assert threshold == torch.tensor(0.7)

