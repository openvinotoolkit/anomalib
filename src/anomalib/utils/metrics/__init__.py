"""Custom anomaly evaluation metrics."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import warnings
from typing import Any

import torchmetrics
from omegaconf import DictConfig, ListConfig

from .anomaly_score_distribution import AnomalyScoreDistribution
from .anomaly_score_threshold import AnomalyScoreThreshold
from .aupr import AUPR
from .aupro import AUPRO
from .auroc import AUROC
from .collection import AnomalibMetricCollection
from .min_max import MinMax
from .optimal_f1 import OptimalF1
from .pro import PRO

__all__ = ["AUROC", "AUPR", "AUPRO", "OptimalF1", "AnomalyScoreThreshold", "AnomalyScoreDistribution", "MinMax", "PRO"]


def metric_collection_from_names(metric_names: list[str], prefix: str | None) -> AnomalibMetricCollection:
    """Create a metric collection from a list of metric names.

    The function will first try to retrieve the metric from the metrics defined in Anomalib metrics module,
    then in TorchMetrics package.

    Args:
        metric_names (list[str]): List of metric names to be included in the collection.
        prefix (str | None): prefix to assign to the metrics in the collection.

    Returns:
        AnomalibMetricCollection: Collection of metrics.
    """
    metrics_module = importlib.import_module("anomalib.utils.metrics")
    metrics = AnomalibMetricCollection([], prefix=prefix)
    for metric_name in metric_names:
        if hasattr(metrics_module, metric_name):
            metric_cls = getattr(metrics_module, metric_name)
            metrics.add_metrics(metric_cls())
        elif hasattr(torchmetrics, metric_name):
            try:
                metric_cls = getattr(torchmetrics, metric_name)
                metrics.add_metrics(metric_cls())
            except TypeError:
                warnings.warn(f"Incorrect constructor arguments for {metric_name} metric from TorchMetrics package.")
        else:
            warnings.warn(f"No metric with name {metric_name} found in Anomalib metrics or TorchMetrics.")
    return metrics


def _validate_metrics_dict(metrics: dict[str, dict[str, Any]]) -> None:
    """Check the assumptions about metrics config dict.

    - Keys are metric names
    - Values are dictionaries.
    - Internal dictionaries:
        - have key "class_path" and its value is of type str
        - have key init_args" and its value is of type dict).

    """
    assert all(
        isinstance(metric, str) for metric in metrics.keys()
    ), f"All keys (metric names) must be strings, found {sorted(metrics.keys())}"
    assert all(
        isinstance(metric, (dict, DictConfig)) for metric in metrics.values()
    ), f"All values must be dictionaries, found {list(metrics.values())}"
    assert all("class_path" in metric and isinstance(metric["class_path"], str) for metric in metrics.values()), (
        "All internal dictionaries must have a 'class_path' key whose value is of type str, "
        f"found {list(metrics.values())}"
    )
    assert all(
        "init_args" in metric and isinstance(metric["init_args"], (dict, DictConfig)) for metric in metrics.values()
    ), (
        "All internal dictionaries must have a 'init_args' key whose value is of type dict, "
        f"found {list(metrics.values())}"
    )


def _get_class_from_path(class_path: str) -> Any:
    """Get a class from a module assuming the string format is `package.subpackage.module.ClassName`."""
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    assert hasattr(module, class_name), f"Class {class_name} not found in module {module_name}"
    cls = getattr(module, class_name)
    return cls


def metric_collection_from_dicts(metrics: dict[str, dict[str, Any]], prefix: str | None) -> AnomalibMetricCollection:
    """Create a metric collection from a dict of "metric name" -> "metric specifications".

    Example:

        metrics = {
            "PixelWiseF1Score": {
                "class_path": "torchmetrics.F1Score",
                "init_args": {},
            },
            "PixelWiseAUROC": {
                "class_path": "anomalib.utils.metrics.AUROC",
                "init_args": {
                    "compute_on_cpu": True,
                },
            },
        }

    In the config file, the same specifications (for pixel-wise metrics) look like:

        ```yaml
        metrics:
            pixel:
                PixelWiseF1Score:
                    class_path: torchmetrics.F1Score
                    init_args: {}
                PixelWiseAUROC:
                    class_path: anomalib.utils.metrics.AUROC
                    init_args:
                        compute_on_cpu: true
        ```

    Args:
        metrics (dict[str, dict[str, Any]]): keys are metric names, values are dictionaries.
            Internal dict[str, Any] keys are "class_path" (value is string) and "init_args" (value is dict),
            following the convention in Pytorch Lightning CLI.

        prefix (str | None): prefix to assign to the metrics in the collection.

    Returns:
        AnomalibMetricCollection: Collection of metrics.
    """
    _validate_metrics_dict(metrics)
    metrics_collection = {}
    for name, dict_ in metrics.items():
        class_path = dict_["class_path"]
        kwargs = dict_["init_args"]
        cls = _get_class_from_path(class_path)
        metrics_collection[name] = cls(**kwargs)
    return AnomalibMetricCollection(metrics_collection, prefix=prefix)


def create_metric_collection(
    metrics: list[str] | dict[str, dict[str, Any]], prefix: str | None
) -> AnomalibMetricCollection:
    """Create a metric collection from a list of metric names or dictionaries.

    This function will dispatch the actual creation to the appropriate function depending on the input type:

        - if list[str] (names of metrics): see `metric_collection_from_names`
        - if dict[str, dict[str, Any]] (path and init args of a class): see `metric_collection_from_dicts`

    The function will first try to retrieve the metric from the metrics defined in Anomalib metrics module,
    then in TorchMetrics package.

    Args:
        metrics (list[str] | dict[str, dict[str, Any]]).
        prefix (str | None): prefix to assign to the metrics in the collection.

    Returns:
        AnomalibMetricCollection: Collection of metrics.
    """
    # fallback is using the names

    if isinstance(metrics, (ListConfig, list)):
        assert all(isinstance(metric, str) for metric in metrics), f"All metrics must be strings, found {metrics}"
        return metric_collection_from_names(metrics, prefix)

    if isinstance(metrics, (DictConfig, dict)):
        _validate_metrics_dict(metrics)
        return metric_collection_from_dicts(metrics, prefix)

    raise ValueError(f"metrics must be a list or a dict, found {type(metrics)}")
