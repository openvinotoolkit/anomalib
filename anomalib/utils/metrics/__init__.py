"""Custom anomaly evaluation metrics."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

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


def get_metrics(config: Union[ListConfig, DictConfig]) -> Tuple[AnomalibMetricCollection, AnomalibMetricCollection]:
    """Create metric collections based on the config.

    Args:
        config (Union[DictConfig, ListConfig]): Config.yaml loaded using OmegaConf

    Returns:
        AnomalibMetricCollection: Image-level metric collection
        AnomalibMetricCollection: Pixel-level metric collection
    """
    image_metric_names = config.metrics.image if "image" in config.metrics.keys() else []
    pixel_metric_names = config.metrics.pixel if "pixel" in config.metrics.keys() else []
    image_metrics = _metric_collection_from_names(image_metric_names, "image_")
    pixel_metrics = _metric_collection_from_names(pixel_metric_names, "pixel_")
    return image_metrics, pixel_metrics


def _metric_collection_from_names(metric_names: List[str], prefix: Optional[str]) -> AnomalibMetricCollection:
    """Create a metric collection from a list of metric names.

    The function will first try to retrieve the metric from the metrics defined in Anomalib metrics module,
    then in TorchMetrics package.

    Args:
        metric_names (List[str]): List of metric names to be included in the collection.
        prefix (Optional[str]): prefix to assign to the metrics in the collection.

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


def _validate_metrics_dict(metrics: Dict[str, Dict[str, Any]]) -> None:
    assert all(
        isinstance(metric, str) for metric in metrics.keys()
    ), f"All keys (metric names) must be strings, found {sorted(metrics.keys())}"
    assert all(
        isinstance(metric, (dict, DictConfig)) for metric in metrics.values()
    ), f"All values must be dictionaries, found {list(metrics.values())}"
    assert all(
        "class_path" in metric and "init_args" for metric in metrics.values()
    ), f"All dictionary must have a 'class_path' and 'kwargs' keys, found {list(metrics.values())}"


def _get_class_from_path(class_path: str) -> Any:
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    assert hasattr(module, class_name), f"Class {class_name} not found in module {module_name}"
    cls = getattr(module, class_name)
    return cls


def _metric_collection_from_dicts(
    metrics: Dict[str, Dict[str, Any]], prefix: Optional[str]
) -> AnomalibMetricCollection:
    """Create a metric collection from a dict of "metric name" -> "metric kwargs".

    The function will first try to retrieve the metric class from `class_path` if this key is present.
    Otherwise the metric is searched first in `anomalib.utils.metrics`, then in `torchmetrics`.

    Args:
        metrics (Dict[str, Dict[str, Any]]): keys are metric names and values are 'kwargs' and 'class_path'.
        prefix (Optional[str]): prefix to assign to the metrics in the collection.

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


def metric_collection_from_names_or_dicts(
    metrics: Union[List[str], Dict[str, Dict[str, Any]]], prefix: Optional[str]
) -> AnomalibMetricCollection:
    """Create a metric collection from a list of metric names or dictionaries.

    - if names: see `metric_collection_from_names`
    - if dicts: see `metric_collection_from_dicts`

    The function will first try to retrieve the metric from the metrics defined in Anomalib metrics module,
    then in TorchMetrics package.

    Args:
        metrics (Union[List[str], Dict[str, Dict[str, Any]]]):
            - if List[str]: metric names to be included in the collection;
            - if Dict[str, Dict[str, Any]]: keys are metric names and values are 'kwargs' and 'class_path'.
        prefix (Optional[str]): prefix to assign to the metrics in the collection.

    Returns:
        AnomalibMetricCollection: Collection of metrics.
    """
    # fallback is using the names

    if isinstance(metrics, (ListConfig, list)):
        assert all(isinstance(metric, str) for metric in metrics), f"All metrics must be strings, found {metrics}"
        return _metric_collection_from_names(metrics, prefix)

    if isinstance(metrics, (DictConfig, dict)):
        _validate_metrics_dict(metrics)
        return _metric_collection_from_dicts(metrics, prefix)

    raise ValueError(f"metrics must be a list or a dict, found {type(metrics)}")
