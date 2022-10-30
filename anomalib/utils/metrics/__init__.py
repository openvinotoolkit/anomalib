"""Custom anomaly evaluation metrics."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
import warnings
from typing import List, Optional, Tuple, Union

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
    image_metrics = metric_collection_from_names(image_metric_names, "image_")
    pixel_metrics = metric_collection_from_names(pixel_metric_names, "pixel_")
    return image_metrics, pixel_metrics


def metric_collection_from_names(metric_names: List[str], prefix: Optional[str]) -> AnomalibMetricCollection:
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
