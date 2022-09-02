"""Implementation of Optimal F1 score based on TorchMetrics."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torchmetrics import Metric, PrecisionRecallCurve


class OptimalF1(Metric):
    """Optimal F1 Metric.

    Compute the optimal F1 score at the adaptive threshold, based on the F1 metric of the true labels and the
    predicted anomaly scores.
    """

    full_state_update: bool = False

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)

        self.precision_recall_curve = PrecisionRecallCurve(num_classes=num_classes)

        self.threshold: torch.Tensor

    # pylint: disable=arguments-differ
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore
        """Update the precision-recall curve metric."""
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
        optimal_f1_score = torch.max(f1_score)
        return optimal_f1_score

    def reset(self) -> None:
        """Reset the metric."""
        self.precision_recall_curve.reset()
