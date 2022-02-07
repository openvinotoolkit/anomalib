"""Implementation of AUROC metric based on TorchMetrics."""
import torch
from torch import Tensor
from torchmetrics import ROC
from torchmetrics.functional import auc


class AUROC(ROC):
    """Area under the ROC curve."""

    def compute(self) -> Tensor:
        """First compute ROC curve, then compute area under the curve.

        Returns:
            Value of the AUROC metric
        """
        tpr: Tensor
        fpr: Tensor

        fpr, tpr, _thresholds = super().compute()
        # TODO: use stable sort after upgrading to pytorch 1.9.x (https://github.com/openvinotoolkit/anomalib/issues/92)
        if not (torch.all(fpr.diff() <= 0) or torch.all(fpr.diff() >= 0)):
            return auc(fpr, tpr, reorder=True)  # only reorder if fpr is not increasing or decreasing
        return auc(fpr, tpr)
