"""Implementation of AUROC metric based on TorchMetrics."""
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
        fpr, tpr, _thresholds = super().compute()
        return auc(fpr, tpr)
