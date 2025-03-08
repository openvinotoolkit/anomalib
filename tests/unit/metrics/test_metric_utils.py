"""Test metric utils."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from jsonargparse import Namespace
from omegaconf import OmegaConf

from anomalib.metrics import AUPRO, MinMax, UnknownMetricError, get_metric

fields = ("pred_score", "gt_label")


class TestGetMetric:
    """Test the `get_metric` method."""

    @staticmethod
    def test_get_metric_placeholder() -> None:
        """Test get_metric with placeholder fields."""
        metric = get_metric("AUPRO", use_placeholder_fields=True)
        assert isinstance(metric, AUPRO)
        assert metric.fields == [""]

    @staticmethod
    def test_get_metric_by_name() -> None:
        """Test get_metric by name."""
        metric = get_metric("AUPRO", use_placeholder_fields=True)
        assert isinstance(metric, AUPRO)
        metric = get_metric("aupro", use_placeholder_fields=True)
        assert isinstance(metric, AUPRO)

        metric = get_metric("MinMax", use_placeholder_fields=True)
        assert isinstance(metric, MinMax)
        metric = get_metric("min_max", use_placeholder_fields=True)
        assert isinstance(metric, MinMax)
        metric = get_metric("minmax", use_placeholder_fields=True)
        assert isinstance(metric, MinMax)

    @staticmethod
    def test_get_metric_by_name_with_init_args() -> None:
        """Test get_metric by name with init args."""
        metric = get_metric("min_max", fields=fields)
        assert isinstance(metric, MinMax)
        assert metric.fields == fields

    @staticmethod
    def test_get_metric_by_dict() -> None:
        """Test get_metric by dict."""
        metric = get_metric({"class_path": "AUPRO"}, use_placeholder_fields=True)
        assert isinstance(metric, AUPRO)

    @staticmethod
    def test_get_metric_by_dict_with_init_args() -> None:
        """Test get_metric by dict with init args."""
        metric = get_metric({"class_path": "MinMax", "init_args": {"fields": fields}})
        assert isinstance(metric, MinMax)
        metric = get_metric({"class_path": "MinMax"}, fields=fields)
        assert isinstance(metric, MinMax)

    @staticmethod
    def test_get_metric_by_dict_with_full_class_path() -> None:
        """Test get_metric by dict with full class path."""
        metric = get_metric({"class_path": "anomalib.metrics.MinMax", "init_args": {"fields": fields}})
        assert isinstance(metric, MinMax)

    @staticmethod
    def test_get_metric_by_namespace() -> None:
        """Test get_metric by namespace."""
        config = OmegaConf.create({"class_path": "MinMax"})
        namespace = Namespace(**config)
        metric = get_metric(namespace, use_placeholder_fields=True)
        assert isinstance(metric, MinMax)

        # Argparse returns an object of type Namespace
        namespace = Namespace(
            class_path="anomalib.metrics.MinMax",
            init_args=Namespace(
                fields=fields,
            ),
        )
        metric = get_metric(namespace)
        assert isinstance(metric, MinMax)

        # Checks the overriding functionality
        metric = get_metric(namespace, fields=fields[::-1])
        assert isinstance(metric, MinMax)
        assert metric.fields == fields[::-1]

    @staticmethod
    def test_get_metric_by_dict_config() -> None:
        """Test get_metric by dict config."""
        config = OmegaConf.create({"class_path": "MinMax"})
        metric = get_metric(config, use_placeholder_fields=True)
        assert isinstance(metric, MinMax)
        config = OmegaConf.create({"class_path": "MinMax", "init_args": {"fields": fields}})
        metric = get_metric(config)
        assert isinstance(metric, MinMax)

    @staticmethod
    def test_get_metric_with_no_fields() -> None:
        """Test get_metric with no fields."""

        def invalid_cases() -> None:
            get_metric("AUPRO")
            get_metric({"class_path": "MinMax"})
            get_metric(OmegaConf.create({"class_path": "MinMax"}))
            get_metric(Namespace(class_path="MinMax"))

        with pytest.raises(ValueError, match="Batch fields must be provided for metric"):
            invalid_cases()

    @staticmethod
    def test_get_unknown_metric() -> None:
        """Test get_metric with unknown metric."""
        with pytest.raises(UnknownMetricError):
            get_metric("UnimplementedMetric")

    @staticmethod
    def test_get_metric_with_invalid_type() -> None:
        """Test get_metric with invalid type."""
        with pytest.raises(TypeError):
            get_metric(OmegaConf.create([{"class_path": "MinMax"}]))

    @staticmethod
    def test_get_metric_with_invalid_class_path() -> None:
        """Test get_metric with invalid class path."""
        with pytest.raises(UnknownMetricError):
            get_metric({"class_path": "anomalib.metrics.InvalidMetric"})
        with pytest.raises(UnknownMetricError):
            get_metric({"class_path": "InvalidMetric"})
        with pytest.raises(UnknownMetricError):
            get_metric({"class_path": "anomalib.typo.InvalidMetric"})
