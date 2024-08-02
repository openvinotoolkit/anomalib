"""Implementation of AUROC metric based on TorchMetrics."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from matplotlib.figure import Figure
from torchmetrics.classification.roc import BinaryROC
from torchmetrics.utilities.compute import auc

from .plotting_utils import plot_figure


class AUROC(BinaryROC):
    """Area under the ROC curve.

    Examples:
        >>> import torch
        >>> from anomalib.metrics import AUROC
        ...
        >>> preds = torch.tensor([0.13, 0.26, 0.08, 0.92, 0.03])
        >>> target = torch.tensor([0, 0, 1, 1, 0])
        ...
        >>> auroc = AUROC()
        >>> auroc(preds, target)
        tensor(0.6667)

        It is possible to update the metric state incrementally:

        >>> auroc.update(preds[:2], target[:2])
        >>> auroc.update(preds[2:], target[2:])
        >>> auroc.compute()
        tensor(0.6667)

        To plot the ROC curve, use the ``generate_figure`` method:

        >>> fig, title = auroc.generate_figure()
    """

    def compute(self) -> torch.Tensor:
        """First compute ROC curve, then compute area under the curve.

        Returns:
            Tensor: Value of the AUROC metric
        """
        tpr: torch.Tensor
        fpr: torch.Tensor

        fpr, tpr = self._compute()
        return auc(fpr, tpr, reorder=True)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with new values.

        Need to flatten new values as ROC expects them in this format for binary classification.

        Args:
            preds (torch.Tensor): predictions of the model
            target (torch.Tensor): ground truth targets
        """
        super().update(preds.flatten(), target.flatten())

    def _compute(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute fpr/tpr value pairs.

        Returns:
            Tuple containing Tensors for fpr and tpr
        """
        tpr: torch.Tensor
        fpr: torch.Tensor
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
