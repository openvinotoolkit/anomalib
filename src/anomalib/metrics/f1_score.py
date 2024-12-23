"""F1 Score and F1Max metrics for binary classification tasks.

This module provides two metrics for evaluating binary classification performance:

- ``F1Score``: Standard F1 score metric that computes the harmonic mean of
  precision and recall at a fixed threshold
- ``F1Max``: Maximum F1 score metric that finds the optimal threshold by
  computing F1 scores across different thresholds

Example:
    >>> from anomalib.metrics import F1Score, F1Max
    >>> import torch
    >>> # Create sample data
    >>> preds = torch.tensor([0.1, 0.4, 0.35, 0.8])
    >>> target = torch.tensor([0, 0, 1, 1])
    >>> # Compute standard F1 score
    >>> f1 = F1Score()
    >>> f1.update(preds > 0.5, target)
    >>> f1.compute()
    tensor(1.0)
    >>> # Compute maximum F1 score
    >>> f1_max = F1Max()
    >>> f1_max.update(preds, target)
    >>> f1_max.compute()
    tensor(1.0)
    >>> f1_max.threshold
    tensor(0.6000)
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torchmetrics import Metric
from torchmetrics.classification import BinaryF1Score

from anomalib.metrics.base import AnomalibMetric

from .precision_recall_curve import BinaryPrecisionRecallCurve


class F1Score(AnomalibMetric, BinaryF1Score):
    """Wrapper to add AnomalibMetric functionality to F1Score metric.

    This class wraps the torchmetrics ``BinaryF1Score`` to make it compatible
    with Anomalib's batch processing capabilities.

    Example:
        >>> from anomalib.metrics import F1Score
        >>> import torch
        >>> # Create metric
        >>> f1 = F1Score()
        >>> # Create sample data
        >>> preds = torch.tensor([0, 0, 1, 1])
        >>> target = torch.tensor([0, 1, 1, 1])
        >>> # Update and compute
        >>> f1.update(preds, target)
        >>> f1.compute()
        tensor(0.8571)
    """


class _F1Max(Metric):
    """F1Max metric for computing the maximum F1 score.

    This class calculates the maximum F1 score by varying the classification
    threshold. The F1 score is the harmonic mean of precision and recall,
    providing a balanced metric for imbalanced datasets.

    After computing the maximum F1 score, the class stores the threshold that
    achieved this score in the ``threshold`` attribute.

    Args:
        **kwargs: Additional arguments passed to the parent ``Metric`` class.

    Attributes:
        full_state_update (bool): Whether to update entire state on each batch.
            Set to ``False`` as metric only needs current batch.
        precision_recall_curve (BinaryPrecisionRecallCurve): Utility to compute
            precision-recall values across thresholds.
        threshold (torch.Tensor): Threshold value that yields maximum F1 score.

    Example:
        >>> from anomalib.metrics import F1Max
        >>> import torch
        >>> # Create metric
        >>> f1_max = F1Max()
        >>> # Create sample data
        >>> preds = torch.tensor([0.1, 0.4, 0.35, 0.8])
        >>> target = torch.tensor([0, 0, 1, 1])
        >>> # Update and compute
        >>> f1_max.update(preds, target)
        >>> f1_max.compute()
        tensor(1.0)
        >>> f1_max.threshold
        tensor(0.6000)
    """

    full_state_update: bool = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.precision_recall_curve = BinaryPrecisionRecallCurve()

        self.threshold: torch.Tensor

    def update(self, preds: torch.Tensor, target: torch.Tensor, *args, **kwargs) -> None:
        """Update the precision-recall curve with new predictions and targets.

        Args:
            preds (torch.Tensor): Predicted scores or probabilities.
            target (torch.Tensor): Ground truth binary labels.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).
        """
        del args, kwargs  # These variables are not used.

        self.precision_recall_curve.update(preds, target)

    def compute(self) -> torch.Tensor:
        """Compute the maximum F1 score across all thresholds.

        Computes F1 scores at different thresholds using the precision-recall
        curve. Stores the threshold that achieves maximum F1 score in the
        ``threshold`` attribute.

        Returns:
            torch.Tensor: Maximum F1 score value.
        """
        precision: torch.Tensor
        recall: torch.Tensor
        thresholds: torch.Tensor

        precision, recall, thresholds = self.precision_recall_curve.compute()
        f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
        self.threshold = thresholds.item() if thresholds.ndim == 0 else thresholds[torch.argmax(f1_score)]
        return torch.max(f1_score)

    def reset(self) -> None:
        """Reset the metric state."""
        self.precision_recall_curve.reset()


class F1Max(AnomalibMetric, _F1Max):  # type: ignore[misc]
    """Wrapper to add AnomalibMetric functionality to F1Max metric.

    This class wraps the internal ``_F1Max`` metric to make it compatible with
    Anomalib's batch processing capabilities.

    Example:
        >>> from anomalib.metrics import F1Max
        >>> from anomalib.data import ImageBatch
        >>> import torch
        >>> # Create metric with batch fields
        >>> f1_max = F1Max(fields=["pred_score", "gt_label"])
        >>> # Create sample batch
        >>> batch = ImageBatch(
        ...     image=torch.rand(4, 3, 32, 32),
        ...     pred_score=torch.tensor([0.1, 0.4, 0.35, 0.8]),
        ...     gt_label=torch.tensor([0, 0, 1, 1])
        ... )
        >>> # Update and compute
        >>> f1_max.update(batch)
        >>> f1_max.compute()
        tensor(1.0)
    """
