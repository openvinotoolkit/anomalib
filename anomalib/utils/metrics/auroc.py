"""Implementation of AUROC metric based on TorchMetrics."""
from typing import Tuple

import torch
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
            Value of the AUROC metric
        """
        tpr: Tensor
        fpr: Tensor

        fpr, tpr = self._compute()
        # TODO: use stable sort after upgrading to pytorch 1.9.x (https://github.com/openvinotoolkit/anomalib/issues/92)
        if not (torch.all(fpr.diff() <= 0) or torch.all(fpr.diff() >= 0)):
            return auc(fpr, tpr, reorder=True)  # only reorder if fpr is not increasing or decreasing
        return auc(fpr, tpr)

    def _compute(self) -> Tuple[Tensor, Tensor]:
        """Method used by Visualizer callback to extract fpr/tpr required for plot generation.

        Returns:
            Tuple containing Tensors for fpr and tpr
        """
        tpr: Tensor
        fpr: Tensor
        fpr, tpr, _thresholds = super().compute()
        return (fpr, tpr)

    def generate_figure(self) -> Tuple[Figure, str]:
        """Generate a figure containing the ROC curve as well as the random baseline and the AUC.

        Returns:
            Tuple[Figure, str]: Tuple containing both the ROC curve and the figure title to be used for logging
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
