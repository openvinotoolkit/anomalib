# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""F1 adaptive threshold metric for anomaly detection.

This module provides the ``F1AdaptiveThreshold`` class which automatically finds
the optimal threshold value by maximizing the F1 score on validation data.

The threshold is computed by:
1. Computing precision-recall curve across multiple thresholds
2. Calculating F1 score at each threshold point
3. Selecting threshold that yields maximum F1 score

Example:
    >>> from anomalib.metrics import F1AdaptiveThreshold
    >>> from anomalib.data import ImageBatch
    >>> import torch
    >>> # Create sample batch
    >>> batch = ImageBatch(
    ...     image=torch.rand(4, 3, 32, 32),
    ...     pred_score=torch.tensor([2.3, 1.6, 2.6, 7.9, 3.3]),
    ...     gt_label=torch.tensor([0, 0, 0, 1, 1])
    ... )
    >>> # Initialize and compute threshold
    >>> threshold = F1AdaptiveThreshold(fields=["pred_score", "gt_label"])
    >>> optimal_threshold = threshold(batch)
    >>> optimal_threshold
    tensor(3.3000)

Note:
    The validation set should contain both normal and anomalous samples for
    reliable threshold computation. A warning is logged if no anomalous samples
    are found.
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager

import torch
from torchmetrics.utilities.data import dim_zero_cat

from anomalib.metrics import AnomalibMetric
from anomalib.metrics.precision_recall_curve import BinaryPrecisionRecallCurve

from .base import Threshold

logger = logging.getLogger(__name__)


@contextmanager
def handle_mac(metric: "_F1AdaptiveThreshold") -> Generator[None, None, None]:
    """Temporarily move tensors to CPU on macOS/MPS and restore after.

    This context manager checks whether the provided metric instance has
    predictions on an MPS device. If so, it moves both predictions and
    targets to CPU for the duration of the context and restores them to
    the original device on exit.

    Note:
        This only applies when ``thresholds=None`` (non-binned mode) where
        ``preds`` and ``target`` lists are used. When thresholds are specified,
        only ``confmat`` is used which doesn't require this handling.
    """
    # Only handle MPS for non-binned mode (thresholds=None) where preds/target exist
    # When thresholds are specified, confmat is used instead
    if metric.thresholds is None and isinstance(metric.preds, list) and metric.preds and metric.preds[0].is_mps:
        original_device = metric.preds[0].device
        metric.preds = [pred.cpu() for pred in metric.preds]
        metric.target = [target.cpu() for target in metric.target]
        try:
            yield
        finally:
            # Restore to original device
            metric.preds = [pred.to(original_device) for pred in metric.preds]
            metric.target = [target.to(original_device) for target in metric.target]
    else:
        yield


class _F1AdaptiveThreshold(BinaryPrecisionRecallCurve, Threshold):
    """Adaptive threshold that maximizes F1 score.

    This class computes and stores the optimal threshold for converting anomaly
    scores to binary predictions by maximizing the F1 score on validation data.

    Example:
        >>> from anomalib.metrics import F1AdaptiveThreshold
        >>> import torch
        >>> # Create validation data
        >>> labels = torch.tensor([0, 0, 1, 1])  # 2 normal, 2 anomalous
        >>> scores = torch.tensor([0.1, 0.2, 0.8, 0.9])  # Anomaly scores
        >>> # Initialize threshold
        >>> threshold = F1AdaptiveThreshold()
        >>> # Compute optimal threshold
        >>> optimal_value = threshold(scores, labels)
        >>> print(f"Optimal threshold: {optimal_value:.4f}")
        Optimal threshold: 0.5000
    """

    def _has_anomalous_samples(self) -> bool:
        """Check if the validation set contains any anomalous samples.

        Returns:
            bool: True if anomalous samples (target=1) exist, False otherwise.
        """
        if self.thresholds is None:
            return any((batch == 1).any().item() for batch in self.target)
        # confmat[i] = [[TN, FP], [FN, TP]]; positives = FN + TP  #  noqa: ERA001
        return (self.confmat[0, 1, 0] + self.confmat[0, 1, 1]).item() > 0

    def _has_normal_samples(self) -> bool:
        """Check if the validation set contains any normal samples.

        Returns:
            bool: True if normal samples (target=0) exist, False otherwise.
        """
        if self.thresholds is None:
            return any((batch == 0).any().item() for batch in self.target)
        # confmat[i] = [[TN, FP], [FN, TP]]; negatives = TN + FP  # noqa: ERA001
        return (self.confmat[0, 0, 0] + self.confmat[0, 0, 1]).item() > 0

    def _get_max_pred(self) -> torch.Tensor:
        """Get the maximum prediction score.

        Returns:
            torch.Tensor: Maximum prediction score. In binned mode, returns the
                highest candidate threshold since raw predictions are unavailable.
        """
        if self.thresholds is None:
            return torch.max(dim_zero_cat(self.preds))
        return self.thresholds[-1]

    def _get_min_pred(self) -> torch.Tensor:
        """Get the minimum prediction score.

        Returns:
            torch.Tensor: Minimum prediction score. In binned mode, returns the
                lowest candidate threshold since raw predictions are unavailable.
        """
        if self.thresholds is None:
            return torch.min(dim_zero_cat(self.preds))
        return self.thresholds[0]

    def compute(self) -> torch.Tensor:
        """Compute optimal threshold by maximizing F1 score.

        Calculates precision-recall curve and corresponding thresholds, then
        finds the threshold that maximizes the F1 score.

        Returns:
            torch.Tensor: Optimal threshold value.

        Warning:
            The fallback behavior when a class is missing depends on whether
            raw predictions are available or binned statistics are used:

            * When raw anomaly scores are available (``self.thresholds`` is
              ``None``), if the validation set contains no anomalous samples,
              the threshold defaults to the maximum anomaly score so that
              normal images are not flagged. If the validation set contains no
              normal samples, the threshold defaults to the minimum anomaly
              score.
            * When only binned statistics are available (``self.thresholds`` is
              not ``None``), raw predictions are not retained and the metric
              instead falls back to the highest or lowest candidate threshold,
              respectively. See :meth:`_get_max_pred` and :meth:`_get_min_pred`.
        """
        precision: torch.Tensor
        recall: torch.Tensor
        thresholds: torch.Tensor

        if not self._has_anomalous_samples():
            if self.thresholds is None:
                fallback = "the highest anomaly score observed in the normal validation images"
            else:
                fallback = "the highest candidate threshold boundary"
            msg = (
                "The validation set does not contain any anomalous images. As a "
                f"result, the adaptive threshold will take the value of {fallback}, "
                "which may lead to poor predictions. For a more reliable "
                "adaptive threshold computation, please add some anomalous "
                "images to the validation set."
            )
            logger.warning(msg)

            self.value = self._get_max_pred()
            return self.value

        if not self._has_normal_samples():
            if self.thresholds is None:
                fallback = "the lowest anomaly score observed in the anomalous validation images"
            else:
                fallback = "the lowest candidate threshold boundary"
            msg = (
                "The validation set does not contain any normal images. As a "
                f"result, the adaptive threshold will take the value of {fallback}, "
                "which may lead to poor predictions. For a more reliable "
                "adaptive threshold computation, please add some normal "
                "images to the validation set."
            )
            logger.warning(msg)

            self.value = self._get_min_pred()
            return self.value

        with handle_mac(self):
            precision, recall, thresholds = super().compute()

        f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
        # NaN arises when precision is 0/0 (no predictions); argmax would select it.
        f1_score = torch.nan_to_num(f1_score, nan=0.0)

        # account for special case where recall is 1.0 even for the highest threshold.
        # In this case 'thresholds' will be scalar.
        return thresholds if thresholds.dim() == 0 else thresholds[torch.argmax(f1_score)]


class F1AdaptiveThreshold(AnomalibMetric, _F1AdaptiveThreshold):  # type: ignore[misc]
    """Wrapper to add AnomalibMetric functionality to F1AdaptiveThreshold metric."""
