"""Implementation of AUROC metric based on TorchMetrics."""
from torch import Tensor
from torchmetrics import ROC
from torchmetrics.functional import auc


class AUROC(ROC):
    """Area under the ROC curve."""

    def __init__(self, force_cpu: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.force_cpu = force_cpu

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update stats for ROC."""
        if self.force_cpu:
            self.cpu()
        super().update(preds.flatten(), target.flatten())

    def compute(self) -> Tensor:
        """First compute ROC curve, then compute area under the curve.

        Returns:
            Value of the AUROC metric
        """
        fpr, tpr, _thresholds = super().compute()
        return auc(fpr, tpr)
