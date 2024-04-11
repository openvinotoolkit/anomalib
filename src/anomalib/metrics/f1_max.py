"""Implementation of F1Max score based on TorchMetrics."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
from torchmetrics import Metric

from anomalib.metrics.precision_recall_curve import BinaryPrecisionRecallCurve

logger = logging.getLogger(__name__)


class F1Max(Metric):
    """F1Max Metric for Computing the Maximum F1 Score.

    This class is designed to calculate the maximum F1 score from the precision-
    recall curve for binary classification tasks. The F1 score is a harmonic
    mean of precision and recall, offering a balance between these two metrics.
    The maximum F1 score (F1-Max) is particularly useful in scenarios where an
    optimal balance between precision and recall is desired, such as in
    imbalanced datasets or when both false positives and false negatives carry
    significant costs.

    After computing the F1Max score, the class also identifies and stores the
    threshold that yields this maximum F1 score, which providing insight into
    the optimal point for the classification decision.

    Args:
        **kwargs: Variable keyword arguments that can be passed to the parent class.

    Attributes:
        full_state_update (bool): Indicates whether the metric requires updating
            the entire state. Set to False for this metric as it calculates the
            F1 score based on the current state without needing historical data.
        precision_recall_curve (BinaryPrecisionRecallCurve): Utility to compute
            precision and recall values across different thresholds.
        threshold (torch.Tensor): Stores the threshold value that results in the
            maximum F1 score.

    Examples:
        >>> from anomalib.metrics import F1Max
        >>> import torch

        >>> preds = torch.tensor([0.1, 0.4, 0.35, 0.8])
        >>> target = torch.tensor([0, 0, 1, 1])

        >>> f1_max = F1Max()
        >>> f1_max.update(preds, target)

        >>> optimal_f1_score = f1_max.compute()
        >>> print(f"Optimal F1 Score: {f1_max_score}")
        >>> print(f"Optimal Threshold: {f1_max.threshold}")

    Note:
        - Use `update` method to input predictions and target labels.
        - Use `compute` method to calculate the maximum F1 score after all
          updates.
        - Use `reset` method to clear the current state and prepare for a new
          set of calculations.
    """

    full_state_update: bool = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.precision_recall_curve = BinaryPrecisionRecallCurve()

        self.threshold: torch.Tensor

    def update(self, preds: torch.Tensor, target: torch.Tensor, *args, **kwargs) -> None:
        """Update the precision-recall curve metric."""
        del args, kwargs  # These variables are not used.

        self.precision_recall_curve.update(preds, target)

    def compute(self) -> torch.Tensor:
        """Compute the value of the optimal F1 score.

        Compute the F1 scores while varying the threshold. Store the optimal
        threshold as attribute and return the maximum value of the F1 score.

        Returns:
            Value of the F1 score at the optimal threshold.
        """
        precision: torch.Tensor
        recall: torch.Tensor
        thresholds: torch.Tensor

        precision, recall, thresholds = self.precision_recall_curve.compute()
        f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
        self.threshold = thresholds[torch.argmax(f1_score)]
        return torch.max(f1_score)

    def reset(self) -> None:
        """Reset the metric."""
        self.precision_recall_curve.reset()
