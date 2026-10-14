# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Area Under the Precision-Recall Curve (AUPR) metric.

This module provides the ``AUPR`` class which computes the area under the
precision-recall curve for evaluating anomaly detection performance.

The AUPR score summarizes the trade-off between precision and recall across
different thresholds. It is particularly useful for imbalanced datasets where
anomalies are rare.

Example:
    >>> from anomalib.metrics import AUPR
    >>> from anomalib.data import ImageBatch
    >>> import torch
    >>> # Create sample batch
    >>> batch = ImageBatch(
    ...     image=torch.rand(4, 3, 32, 32),
    ...     pred_score=torch.tensor([0.1, 0.2, 0.8, 0.9]),
    ...     gt_label=torch.tensor([0, 0, 1, 1])
    ... )
    >>> # Initialize and compute AUPR
    >>> aupr = AUPR(fields=["pred_score", "gt_label"])
    >>> aupr(batch)
    tensor(1.)

The metric can also be updated incrementally:

    >>> for batch in dataloader:
    ...     aupr.update(batch)
    >>> final_score = aupr.compute()

Note:
    The AUPR score ranges from 0 to 1, with 1 indicating perfect ranking of
    anomalies above normal samples. A detector whose scores carry no ranking
    information scores the fraction of anomalous samples.
"""

import torch
from matplotlib.figure import Figure
from torchmetrics.classification import BinaryPrecisionRecallCurve
from torchmetrics.utilities.data import dim_zero_cat

from .base import AnomalibMetric
from .utils import plot_metric_curve


class _AUPR(BinaryPrecisionRecallCurve):
    """Area under the PR curve.

    This metric computes the area under the precision-recall curve as step-wise
    average precision.

    Args:
        kwargs: Additional arguments to the TorchMetrics base class.

    Examples:
        To compute the metric for a set of predictions and ground truth targets:
        >>> from anomalib.metrics.aupr import _AUPR
        >>> import torch
        >>> true = torch.tensor([0, 1, 1, 1, 0, 0, 0, 0, 1, 1])
        >>> pred = torch.tensor([0.59, 0.35, 0.72, 0.33, 0.73, 0.81, 0.30, 0.05, 0.04, 0.48])

        >>> metric = _AUPR()
        >>> metric(pred, true)
        tensor(0.4610)

        It is also possible to update the metric state incrementally within batches:

        >>> for pred, true in dataloader:
        ...     # Compute prediction and target tensors
        ...     metric.update(pred, true)
        >>> metric.compute()

        Once the metric has been computed, we can plot the PR curve:

        >>> figure, title = metric.generate_figure()
    """

    def compute(self) -> torch.Tensor:
        """First compute PR curve, then compute area under the curve.

        The area is the step-wise average precision. Trapezoidal integration of the
        curve sorted by recall does not give the area under a precision-recall curve,
        and overstates it most for detectors with few distinct scores.

        Returns:
            Value of the AUPR metric
        """
        prec: torch.Tensor
        rec: torch.Tensor

        prec, rec = self._compute()
        prec = torch.where(torch.isnan(prec), torch.zeros_like(prec), prec)
        rec = torch.where(torch.isnan(rec), torch.zeros_like(rec), rec)
        # Recall is in decreasing order, so the differences are negative.
        return -torch.sum((rec[1:] - rec[:-1]) * prec[:-1])

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
        xlabel = "Recall"
        ylabel = "Precision"
        loc = "best"
        title = "PR"

        fig, axis = plot_metric_curve(rec, prec, aupr, xlim, ylim, xlabel, ylabel, loc, title, metric_name="AUPR")

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
