"""Area Under the Receiver Operating Characteristic (AUROC) metric.

This module provides the ``AUROC`` class which computes the area under the ROC
curve for evaluating anomaly detection performance.

The AUROC score summarizes the trade-off between true positive rate (TPR) and
false positive rate (FPR) across different thresholds. It measures how well the
model can distinguish between normal and anomalous samples.

Example:
    >>> from anomalib.metrics import AUROC
    >>> import torch
    >>> # Create sample data
    >>> labels = torch.tensor([0, 0, 1, 1])  # Binary labels
    >>> scores = torch.tensor([0.1, 0.2, 0.8, 0.9])  # Anomaly scores
    >>> # Initialize and compute AUROC
    >>> metric = AUROC()
    >>> auroc_score = metric(scores, labels)
    >>> auroc_score
    tensor(1.0)

The metric can also be updated incrementally with batches:

    >>> for batch_scores, batch_labels in dataloader:
    ...     metric.update(batch_scores, batch_labels)
    >>> final_score = metric.compute()

Once computed, the ROC curve can be visualized:

    >>> figure, title = metric.generate_figure()

Note:
    The AUROC score ranges from 0 to 1, with 1 indicating perfect ranking of
    anomalies above normal samples.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from matplotlib.figure import Figure
from torchmetrics.classification.roc import BinaryROC
from torchmetrics.utilities.compute import auc

from anomalib.metrics.base import AnomalibMetric

from .plotting_utils import plot_figure


class _AUROC(BinaryROC):
    """Area under the ROC curve.

    This class computes the area under the receiver operating characteristic
    curve, which plots the true positive rate against the false positive rate
    at various thresholds.

    Examples:
        To compute the metric for a set of predictions and ground truth targets:

        >>> import torch
        >>> from anomalib.metrics import AUROC
        >>> preds = torch.tensor([0.13, 0.26, 0.08, 0.92, 0.03])
        >>> target = torch.tensor([0, 0, 1, 1, 0])
        >>> auroc = AUROC()
        >>> auroc(preds, target)
        tensor(0.6667)

        It is possible to update the metric state incrementally:

        >>> auroc.update(preds[:2], target[:2])
        >>> auroc.update(preds[2:], target[2:])
        >>> auroc.compute()
        tensor(0.6667)

        To plot the ROC curve:

        >>> figure, title = auroc.generate_figure()
    """

    def compute(self) -> torch.Tensor:
        """First compute ROC curve, then compute area under the curve.

        Returns:
            torch.Tensor: Value of the AUROC metric
        """
        tpr: torch.Tensor
        fpr: torch.Tensor

        fpr, tpr = self._compute()
        return auc(fpr, tpr, reorder=True)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with new predictions and targets.

        Need to flatten new values as ROC expects them in this format for binary
        classification.

        Args:
            preds (torch.Tensor): Predictions from the model
            target (torch.Tensor): Ground truth target labels
        """
        super().update(preds.flatten(), target.flatten())

    def _compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute false positive rate and true positive rate value pairs.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing tensors for FPR
                and TPR values
        """
        tpr: torch.Tensor
        fpr: torch.Tensor
        fpr, tpr, _thresholds = super().compute()
        return (fpr, tpr)

    def generate_figure(self) -> tuple[Figure, str]:
        """Generate a figure showing the ROC curve.

        The figure includes the ROC curve, a baseline representing random
        performance, and the AUROC score.

        Returns:
            tuple[Figure, str]: Tuple containing both the figure and the figure
                title to be used for logging
        """
        fpr, tpr = self._compute()
        auroc = self.compute()

        xlim = (0.0, 1.0)
        ylim = (0.0, 1.0)
        xlabel = "False Positive Rate"
        ylabel = "True Positive Rate"
        loc = "lower right"
        title = "ROC"

        fig, axis = plot_figure(fpr, tpr, auroc, xlim, ylim, xlabel, ylabel, loc, title)

        axis.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=2,
            linestyle="--",
            figure=fig,
        )

        return fig, title


class AUROC(AnomalibMetric, _AUROC):  # type: ignore[misc]
    """Wrapper to add AnomalibMetric functionality to AUROC metric."""
