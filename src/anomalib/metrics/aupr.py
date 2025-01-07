"""Area Under the Precision-Recall Curve (AUPR) metric.

This module provides the ``AUPR`` class which computes the area under the
precision-recall curve for evaluating anomaly detection performance.

The AUPR score summarizes the trade-off between precision and recall across
different thresholds. It is particularly useful for imbalanced datasets where
anomalies are rare.

Example:
    >>> from anomalib.metrics import AUPR
    >>> import torch
    >>> # Create sample data
    >>> labels = torch.tensor([0, 0, 1, 1])  # Binary labels
    >>> scores = torch.tensor([0.1, 0.2, 0.8, 0.9])  # Anomaly scores
    >>> # Initialize and compute AUPR
    >>> metric = AUPR()
    >>> aupr_score = metric(scores, labels)
    >>> aupr_score
    tensor(1.0)

The metric can also be updated incrementally with batches:

    >>> for batch_scores, batch_labels in dataloader:
    ...     metric.update(batch_scores, batch_labels)
    >>> final_score = metric.compute()

Note:
    The AUPR score ranges from 0 to 1, with 1 indicating perfect ranking of
    anomalies above normal samples.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from matplotlib.figure import Figure
from torchmetrics.classification import BinaryPrecisionRecallCurve
from torchmetrics.utilities.compute import auc
from torchmetrics.utilities.data import dim_zero_cat

from .base import AnomalibMetric
from .plotting_utils import plot_figure


class _AUPR(BinaryPrecisionRecallCurve):
    """Area under the PR curve.

    This metric computes the area under the precision-recall curve.

    Args:
        kwargs: Additional arguments to the TorchMetrics base class.

    Examples:
        To compute the metric for a set of predictions and ground truth targets:

        >>> true = torch.tensor([0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
        >>> pred = torch.tensor([0.59, 0.35, 0.72, 0.33, 0.73, 0.81, 0.30, 0.05, 0.04, 0.48])

        >>> metric = AUPR()
        >>> metric(pred, true)
        tensor(0.4899)

        It is also possible to update the metric state incrementally within batches:

        >>> for batch in dataloader:
        ...     # Compute prediction and target tensors
        ...     metric.update(pred, true)
        >>> metric.compute()

        Once the metric has been computed, we can plot the PR curve:

        >>> figure, title = metric.generate_figure()
    """

    def compute(self) -> torch.Tensor:
        """First compute PR curve, then compute area under the curve.

        Returns:
            Value of the AUPR metric
        """
        prec: torch.Tensor
        rec: torch.Tensor

        prec, rec = self._compute()
        return auc(rec, prec, reorder=True)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with new values.

        Need to flatten new values as PrecicionRecallCurve expects them in this format for binary classification.

        Args:
            preds (torch.Tensor): predictions of the model
            target (torch.Tensor): ground truth targets
        """
        super().update(preds.flatten(), target.flatten())

    def _compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute prec/rec value pairs.

        Returns:
            Tuple containing Tensors for rec and prec
        """
        prec: torch.Tensor
        rec: torch.Tensor
        prec, rec, _ = super().compute()
        return (prec, rec)

    def generate_figure(self) -> tuple[Figure, str]:
        """Generate a figure containing the PR curve as well as the random baseline and the AUC.

        Returns:
            tuple[Figure, str]: Tuple containing both the PR curve and the figure title to be used for logging
        """
        prec, rec = self._compute()
        aupr = self.compute()

        xlim = (0.0, 1.0)
        ylim = (0.0, 1.0)
        xlabel = "Precision"
        ylabel = "Recall"
        loc = "best"
        title = "AUPR"

        fig, axis = plot_figure(rec, prec, aupr, xlim, ylim, xlabel, ylabel, loc, title)

        # Baseline in PR-curve is the prevalence of the positive class
        rate = (dim_zero_cat(self.target) == 1).sum() / (dim_zero_cat(self.target).size(0))
        axis.plot(
            (0, 1),
            (rate.detach().cpu(), rate.detach().cpu()),
            color="navy",
            lw=2,
            linestyle="--",
            figure=fig,
        )

        return fig, title


class AUPR(AnomalibMetric, _AUPR):  # type: ignore[misc]
    """Wrapper to add AnomalibMetric functionality to AUPR metric."""
