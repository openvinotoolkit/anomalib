"""Implementation of AUROC metric based on TorchMetrics."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics import PrecisionRecallCurve
from torchmetrics.functional import auc
from torchmetrics.utilities.data import dim_zero_cat

from .plotting_utils import plot_figure


class AUPR(PrecisionRecallCurve):
    """Area under the PR curve."""

    def compute(self) -> Tensor:
        """First compute PR curve, then compute area under the curve.

        Returns:
            Value of the AUPR metric
        """
        prec: Tensor
        rec: Tensor

        prec, rec = self._compute()
        return auc(rec, prec, reorder=True)

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with new values.

        Need to flatten new values as PrecicionRecallCurve expects them in this format for binary classification.

        Args:
            preds (Tensor): predictions of the model
            target (Tensor): ground truth targets
        """
        super().update(preds.flatten(), target.flatten())

    def _compute(self) -> tuple[Tensor, Tensor]:
        """Compute prec/rec value pairs.

        Returns:
            Tuple containing Tensors for rec and prec
        """
        prec: Tensor
        rec: Tensor
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
