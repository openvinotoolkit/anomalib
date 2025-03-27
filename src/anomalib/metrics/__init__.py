"""Custom metrics for evaluating anomaly detection models.

This module provides various metrics for evaluating anomaly detection performance:

- Area Under Curve (AUC) metrics:
    - ``AUROC``: Area Under Receiver Operating Characteristic curve
    - ``AUPR``: Area Under Precision-Recall curve
    - ``AUPRO``: Area Under Per-Region Overlap curve
    - ``AUPIMO``: Area Under Per-Image Missed Overlap curve

- F1-score metrics:
    - ``F1Score``: Standard F1 score
    - ``F1Max``: Maximum F1 score across thresholds

- Threshold metrics:
    - ``F1AdaptiveThreshold``: Finds optimal threshold by maximizing F1 score
    - ``ManualThreshold``: Uses manually specified threshold

- Other metrics:
    - ``AnomalibMetric``: Base class for custom metrics
    - ``AnomalyScoreDistribution``: Analyzes score distributions
    - ``BinaryPrecisionRecallCurve``: Computes precision-recall curves
    - ``Evaluator``: Combines multiple metrics for evaluation
    - ``MinMax``: Normalizes scores to [0,1] range
    - ``PRO``: Per-Region Overlap score
    - ``PIMO``: Per-Image Missed Overlap score

Example:
    >>> from anomalib.metrics import AUROC, F1Score
    >>> auroc = AUROC()
    >>> f1 = F1Score()
    >>> labels = torch.tensor([0, 1, 0, 1])
    >>> scores = torch.tensor([0.1, 0.9, 0.2, 0.8])
    >>> auroc(scores, labels)
    tensor(1.)
    >>> f1(scores, labels, threshold=0.5)
    tensor(1.)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from importlib import import_module

from jsonargparse import Namespace
from omegaconf import DictConfig, OmegaConf

from anomalib.utils.path import convert_to_snake_case

from .anomaly_score_distribution import AnomalyScoreDistribution
from .aupr import AUPR
from .aupro import AUPRO
from .auroc import AUROC
from .base import AnomalibMetric, create_anomalib_metric
from .evaluator import Evaluator
from .f1_score import F1Max, F1Score
from .min_max import MinMax
from .pimo import AUPIMO, PIMO
from .precision_recall_curve import BinaryPrecisionRecallCurve
from .pro import PRO
from .threshold import F1AdaptiveThreshold, ManualThreshold

__all__ = [
    "AUROC",
    "AUPR",
    "AUPRO",
    "AnomalibMetric",
    "AnomalyScoreDistribution",
    "BinaryPrecisionRecallCurve",
    "create_anomalib_metric",
    "Evaluator",
    "F1AdaptiveThreshold",
    "F1Max",
    "F1Score",
    "ManualThreshold",
    "MinMax",
    "PRO",
    "PIMO",
    "AUPIMO",
]


class UnknownMetricError(ModuleNotFoundError):
    pass


logger = logging.getLogger(__name__)


def get_available_metrics() -> set[str]:
    """Get set of available anomaly detection metrics.

    Returns a set of metric names in snake_case format from the Anomalib library.
    However, fully uppercase metric names, such as ``'AUPRO'``, are returned in lowercase.

    Returns:
        set[str]: Set of available metric names in snake_case format (e.g.
            ``'min_max'``, ``'auroc'``, etc.)

    Example:
        Get all available metrics:

        >>> from anomalib.metrics import get_available_metrics
        >>> metrics = get_available_metrics()
        >>> print(sorted(list(metrics)))  # doctest: +NORMALIZE_WHITESPACE
            ['aupimo', 'aupr', 'aupro', 'auroc',
             'f1_adaptive_threshold', 'f1_max', 'f1_score',
             'min_max', 'pimo', 'pro']

    Note:
        The returned metric names can be used with :func:`get_metric` to instantiate
        the corresponding metrics class.
    """
    return {
        cls.__name__.lower() if cls.__name__.isupper() else convert_to_snake_case(cls.__name__)
        for cls in AnomalibMetric.__subclasses__()
        if cls.__name__ != "AnomalibMetric"
    }


def _get_metric_class_by_name(name: str) -> type[AnomalibMetric]:
    """Retrieve an anomaly metric class based on its name.

    This internal function takes a metric name and returns the corresponding metric class.
    The name matching is case-insensitive and supports both snake_case and PascalCase
    formats.

    Args:
        name (str): Name of the metric to retrieve. Can be in snake_case (e.g.
            ``"min_max"``) or PascalCase (e.g. ``"MinMax"``). The name is
            case-insensitive.

    Raises:
        UnknownMetricError: If no metric is found matching the provided name. The error
            message includes the list of available metrics.

    Returns:
        type[AnomalibMetric]: Metric class that inherits from ``AnomalibMetric``.

    Examples:
        >>> from anomalib.metrics import _get_metric_class_by_name
        >>> metric_class = _get_metric_class_by_name("auroc")
        >>> metric_class.__name__
        'AUROC'
        >>> metric_class = _get_metric_class_by_name("min_max")
        >>> metric_class.__name__
        'MinMax'
    """
    from anomalib.models import convert_snake_to_pascal_case

    logger.info("Loading the metric..")
    metric_class: type[AnomalibMetric] | None = None

    name = convert_snake_to_pascal_case(name).lower() if "_" in name else name.lower()
    for metric in AnomalibMetric.__subclasses__():
        if name == metric.__name__.lower():
            metric_class = metric
    if metric_class is None:
        logger.exception(f"Could not find the metric {name}. Available metric are {get_available_metrics()}")
        raise UnknownMetricError

    return metric_class


def get_metric(
    metric: DictConfig | str | dict | Namespace,
    use_placeholder_fields: bool = False,
    *args,
    **kwdargs,
) -> AnomalibMetric:
    """Get an anomaly detection metric instance.

    This function instantiates an anomaly detection metric based on the provided
    configuration or metric name. It supports multiple ways of metric specification
    including string names, dictionaries and OmegaConf configurations.

    Args:
        metric (DictConfig | str | dict | Namespace): Metric specification that can be:
            - A string with metric name (e.g. ``"min_max"``, ``"auroc"``)
            - A dictionary with ``class_path`` and optional ``init_args``
            - An OmegaConf DictConfig with similar structure as dict
            - A Namespace object with similar structure as dict
        use_placeholder_fields (bool): If `True` and `fields` is not provided as
            the first argument in `args` or as a key in `kwargs`,
            it initializes `fields` with a list containing an empty string as a
            placeholder.
        *args: Variable length argument list passed to metric initialization.
        **kwdargs: Arbitrary keyword arguments passed to metric initialization.

    Returns:
        AnomalibMetric: Instantiated anomaly detection metric.

    Raises:
        TypeError: If ``metric`` argument is of unsupported type.
        UnknownMetricError: If specified metric class cannot be found.

    Examples:
        Get metric by name:

        >>> metric = get_metric("min_max", use_placeholder_fields=True)
        >>> metric = get_metric("f1_score", use_placeholder_fields=True)
        >>> metric = get_metric("auroc", fields=("pred_labels", "gt_labels"))

        Get metric using dictionary config:

        >>> metric = get_metric({"class_path": "AUPRO"})
        >>> metric = get_metric(
        ...     {"class_path": "MinMax"},
        ...     fields=("pred_labels", "gt_labels")
        ... )
        >>> metric = get_metric({
        ...     "class_path": "F1Score",
        ...     "init_args": {"fields": ("pred_labels", "gt_labels")}
        ... })

        Get metric using fully qualified path:

        >>> metric = get_metric({
        ...     "class_path": "anomalib.metrics.F1Score",
        ...     "init_args": {"fields": ("pred_labels", "gt_labels")}
        ... })
    """
    _metric: AnomalibMetric
    if isinstance(metric, str):
        _metric_class = _get_metric_class_by_name(metric)
        if use_placeholder_fields and not (len(args) or kwdargs.get("fields", None)):
            logger.warning("Initializing the metrics with empty fields parameter which may raise exception.")
            kwdargs["fields"] = [""]
        _metric = _metric_class(*args, **kwdargs)
    elif isinstance(metric, DictConfig | Namespace | dict):
        if isinstance(metric, dict):
            metric = OmegaConf.create(metric)
        path_split = metric.class_path.rsplit(".", 1)
        try:
            module = import_module(path_split[0]) if len(path_split) > 1 else import_module("anomalib.metrics")
        except ModuleNotFoundError as exception:
            logger.exception(
                f"Could not find the module {metric.class_path}. Available metrics are {get_available_metrics()}",
            )
            raise UnknownMetricError from exception
        try:
            metric_class = getattr(module, path_split[-1])
            init_args = metric.get("init_args", {})
            if isinstance(init_args, Namespace):
                for key, value in kwdargs.items():
                    init_args.update(value, key)
            else:
                init_args.update(kwdargs)
            if use_placeholder_fields and not (len(args) or init_args.get("fields", None)):
                logger.warning("Initializing the metrics with empty fields parameter which may raise exception.")
                init_args["fields"] = [""]
            _metric = metric_class(*args, **init_args)
        except AttributeError as exception:
            logger.exception(
                f"Could not find the metric {metric.class_path}. Available metrics are {get_available_metrics()}",
            )
            raise UnknownMetricError from exception
    else:
        logger.error(f"Unsupported type {type(metric)} for metric configuration.")
        raise TypeError
    return _metric
