"""Implementation of AUROC metric based on TorchMetrics."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics import ROC
from torchmetrics.functional import auc

from .plotting_utils import plot_figure


class AUROC(ROC):
    """Area under the ROC curve."""

    def compute(self) -> Tensor:
        """First compute ROC curve, then compute area under the curve.

        Returns:
            Tensor: Value of the AUROC metric
        """
        tpr: Tensor
        fpr: Tensor

        fpr, tpr = self._compute()
        return auc(fpr, tpr, reorder=True)

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with new values.

        Need to flatten new values as ROC expects them in this format for binary classification.

        Args:
            preds (Tensor): predictions of the model
            target (Tensor): ground truth targets
        """
        super().update(preds.flatten(), target.flatten())

    def _compute(self) -> tuple[Tensor, Tensor]:
        """Compute fpr/tpr value pairs.

        Returns:
            Tuple containing Tensors for fpr and tpr
        """
        tpr: Tensor
        fpr: Tensor
        fpr, tpr, _thresholds = super().compute()
        return (fpr, tpr)

    def generate_figure(self) -> tuple[Figure, str]:
        """Generate a figure containing the ROC curve, the baseline and the AUROC.

        Returns:
            tuple[Figure, str]: Tuple containing both the figure and the figure title to be used for logging
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
