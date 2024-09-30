"""Custom anomaly evaluation metrics."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
import logging
from collections.abc import Callable
from typing import Any

import torchmetrics
from omegaconf import DictConfig, ListConfig

from .anomaly_score_distribution import AnomalyScoreDistribution
from .aupr import AUPR
from .aupro import AUPRO
from .auroc import AUROC
from .collection import AnomalibMetricCollection
from .f1_max import F1Max
from .f1_score import F1Score
from .min_max import MinMax
from .pimo import AUPIMO, PIMO
from .precision_recall_curve import BinaryPrecisionRecallCurve
from .pro import PRO
from .threshold import F1AdaptiveThreshold, ManualThreshold

__all__ = [
    "AUROC",
    "AUPR",
    "AUPRO",
    "AnomalyScoreDistribution",
    "BinaryPrecisionRecallCurve",
    "F1AdaptiveThreshold",
    "F1Max",
    "F1Score",
    "ManualThreshold",
    "MinMax",
    "PRO",
    "PIMO",
    "AUPIMO",
]

logger = logging.getLogger(__name__)


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
    metrics_module = importlib.import_module("anomalib.metrics")
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
                msg = f"Incorrect constructor arguments for {metric_name} metric from TorchMetrics package."
                logger.warning(msg)
        else:
            msg = f"No metric with name {metric_name} found in Anomalib metrics or TorchMetrics."
            logger.warning(msg)
    return metrics


def _validate_metrics_dict(metrics: dict[str, dict[str, Any]]) -> None:
    """Check the assumptions about metrics config dict.

    - Keys are metric names
    - Values are dictionaries.
    - Internal dictionaries:
        - have key "class_path" and its value is of type str
        - have key init_args" and its value is of type dict).

    """
    if not all(isinstance(metric, str) for metric in metrics):
        msg = f"All keys (metric names) must be strings, found {sorted(metrics.keys())}"
        raise TypeError(msg)

    if not all(isinstance(metric, DictConfig | dict) for metric in metrics.values()):
        msg = f"All values must be dictionaries, found {list(metrics.values())}"
        raise TypeError(msg)

    if not all("class_path" in metric and isinstance(metric["class_path"], str) for metric in metrics.values()):
        msg = "All internal dictionaries must have a 'class_path' key whose value is of type str."
        raise ValueError(msg)

    if not all(
        "init_args" in metric and isinstance(metric["init_args"], dict) or isinstance(metric["init_args"], DictConfig)
        for metric in metrics.values()
    ):
        msg = "All internal dictionaries must have a 'init_args' key whose value is of type dict."
        raise ValueError(msg)


def _get_class_from_path(class_path: str) -> Callable:
    """Get a class from a module assuming the string format is `package.subpackage.module.ClassName`."""
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    if not hasattr(module, class_name):
        msg = f"Class {class_name} not found in module {module_name}"
        raise AttributeError(msg)
    return getattr(module, class_name)


def metric_collection_from_dicts(metrics: dict[str, dict[str, Any]], prefix: str | None) -> AnomalibMetricCollection:
    """Create a metric collection from a dict of "metric name" -> "metric specifications".

    Example:
        metrics = {
            "PixelWiseF1Score": {
                "class_path": "torchmetrics.F1Score",
                "init_args": {},
            },
            "PixelWiseAUROC": {
                "class_path": "anomalib.metrics.AUROC",
                "init_args": {
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
                    class_path: anomalib.metrics.AUROC

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
    metrics: list[str] | dict[str, dict[str, Any]],
    prefix: str | None = None,
) -> AnomalibMetricCollection:
    """Create a metric collection from a list of metric names or dictionaries.

    This function will dispatch the actual creation to the appropriate function depending on the input type:

        - if list[str] (names of metrics): see `metric_collection_from_names`
        - if dict[str, dict[str, Any]] (path and init args of a class): see `metric_collection_from_dicts`

    The function will first try to retrieve the metric from the metrics defined in Anomalib metrics module,
    then in TorchMetrics package.

    Args:
        metrics (list[str] | dict[str, dict[str, Any]]): List of metrics or dictionaries to create metric collection.
        prefix (str | None): Prefix to assign to the metrics in the collection.

    Returns:
        AnomalibMetricCollection: Collection of metrics.
    """
    # fallback is using the names

    if isinstance(metrics, ListConfig | list):
        if not all(isinstance(metric, str) for metric in metrics):
            msg = f"All metrics must be strings, found {metrics}"
            raise TypeError(msg)

        return metric_collection_from_names(metrics, prefix)

    if isinstance(metrics, DictConfig | dict):
        _validate_metrics_dict(metrics)
        return metric_collection_from_dicts(metrics, prefix)

    msg = f"metrics must be a list or a dict, found {type(metrics)}"
    raise ValueError(msg)
