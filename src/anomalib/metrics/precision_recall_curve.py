"""Custom implementation of Precision-Recall Curve metric.

This module provides a custom implementation of the binary precision-recall curve
metric that does not apply sigmoid normalization to prediction thresholds, unlike
the standard torchmetrics implementation.

See: https://github.com/Lightning-AI/torchmetrics/issues/1526

Example:
    >>> import torch
    >>> from anomalib.metrics import BinaryPrecisionRecallCurve
    >>> # Create sample predictions and targets
    >>> preds = torch.tensor([0.1, 0.4, 0.35, 0.8])
    >>> target = torch.tensor([0, 0, 1, 1])
    >>> # Initialize metric
    >>> pr_curve = BinaryPrecisionRecallCurve()
    >>> # Update metric state
    >>> pr_curve.update(preds, target)
    >>> # Compute precision, recall and thresholds
    >>> precision, recall, thresholds = pr_curve.compute()
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from torch import Tensor
from torchmetrics.classification import BinaryPrecisionRecallCurve as _BinaryPrecisionRecallCurve
from torchmetrics.functional.classification.precision_recall_curve import (
    _adjust_threshold_arg,
    _binary_precision_recall_curve_update,
)


class BinaryPrecisionRecallCurve(_BinaryPrecisionRecallCurve):
    """Binary precision-recall curve without threshold prediction normalization.

    This class extends the torchmetrics ``BinaryPrecisionRecallCurve`` class but
    removes the sigmoid normalization step applied to prediction thresholds.

    Example:
        >>> import torch
        >>> from anomalib.metrics import BinaryPrecisionRecallCurve
        >>> metric = BinaryPrecisionRecallCurve()
        >>> preds = torch.tensor([0.1, 0.4, 0.35, 0.8])
        >>> target = torch.tensor([0, 0, 1, 1])
        >>> metric.update(preds, target)
        >>> precision, recall, thresholds = metric.compute()
    """

    @staticmethod
    def _binary_precision_recall_curve_format(
        preds: Tensor,
        target: Tensor,
        thresholds: int | list[float] | Tensor | None = None,
        ignore_index: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Format predictions and targets for binary precision-recall curve.

        Similar to torchmetrics' ``_binary_precision_recall_curve_format`` but
        without sigmoid normalization of predictions.

        Args:
            preds (Tensor): Predicted scores or probabilities
            target (Tensor): Ground truth binary labels
            thresholds (int | list[float] | Tensor | None, optional): Thresholds
                used for computing curve points. Defaults to ``None``.
            ignore_index (int | None, optional): Label to ignore in evaluation.
                Defaults to ``None``.

        Returns:
            tuple[Tensor, Tensor, Tensor | None]: Tuple containing:
                - Flattened predictions
                - Flattened targets
                - Adjusted thresholds
        """
        preds = preds.flatten()
        target = target.flatten()
        if ignore_index is not None:
            idx = target != ignore_index
            preds = preds[idx]
            target = target[idx]

        thresholds = _adjust_threshold_arg(thresholds, preds.device)
        return preds, target, thresholds

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric state with new predictions and targets.

        Unlike the base class, this method accepts raw predictions without
        applying sigmoid normalization.

        Args:
            preds (Tensor): Raw predicted scores or probabilities
            target (Tensor): Ground truth binary labels (0 or 1)
        """
        preds, target, _ = BinaryPrecisionRecallCurve._binary_precision_recall_curve_format(
            preds,
            target,
            self.thresholds,
            self.ignore_index,
        )
        state = _binary_precision_recall_curve_update(preds, target, self.thresholds)
        if isinstance(state, Tensor):
            self.confmat += state
        else:
            self.preds.append(state[0])
            self.target.append(state[1])
