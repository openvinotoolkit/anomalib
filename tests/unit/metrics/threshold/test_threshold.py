"""Test Threshold metric."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from torchmetrics import Metric

from anomalib.metrics.threshold import BaseThreshold, Threshold


class TestThreshold:
    """Test cases for the Threshold class."""

    @staticmethod
    def test_threshold_abstract_methods() -> None:
        """Test that Threshold class raises NotImplementedError for abstract methods."""
        threshold = Threshold()

        with pytest.raises(NotImplementedError, match="Subclass of Threshold must implement the compute method"):
            threshold.compute()

        with pytest.raises(NotImplementedError, match="Subclass of Threshold must implement the update method"):
            threshold.update()

    @staticmethod
    def test_threshold_initialization() -> None:
        """Test that Threshold can be initialized without errors."""
        threshold = Threshold()
        assert isinstance(threshold, Metric)


class TestBaseThreshold:
    """Test cases for the BaseThreshold class."""

    @staticmethod
    def test_base_threshold_deprecation_warning() -> None:
        """Test that BaseThreshold class raises a DeprecationWarning."""
        with pytest.warns(
            DeprecationWarning,
            match="BaseThreshold is deprecated and will be removed in a future version. Use Threshold instead.",
        ):
            BaseThreshold()

    @staticmethod
    def test_base_threshold_inheritance() -> None:
        """Test that BaseThreshold inherits from Threshold."""
        base_threshold = BaseThreshold()
        assert isinstance(base_threshold, Threshold)

    @staticmethod
    def test_base_threshold_abstract_methods() -> None:
        """Test that BaseThreshold class raises NotImplementedError for abstract methods."""
        base_threshold = BaseThreshold()

        with pytest.raises(NotImplementedError, match="Subclass of Threshold must implement the compute method"):
            base_threshold.compute()

        with pytest.raises(NotImplementedError, match="Subclass of Threshold must implement the update method"):
            base_threshold.update()
