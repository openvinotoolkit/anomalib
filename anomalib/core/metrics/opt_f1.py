"""
Implementation of Optimal F1 score based on TorchMetrics.
"""
import torch
from torch import Tensor
from torchmetrics import PrecisionRecallCurve


class OptimalF1(PrecisionRecallCurve):
    """
    Compute the optimal F1 score at the adaptive threshold, based on the F1 metric of the
    true labels and the predicted anomaly scores.
    """

    threshold: torch.Tensor

    def compute(self) -> Tensor:
        """
        Compute the F1 scores while varying the threshold. Store the optimal
        threshold as attribute and return the maximum value of the F1 score.

        Returns:
            Value of the F1 score at the optimal threshold.
        """
        precision, recall, thresholds = super().compute()
        f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
        self.threshold = thresholds[torch.argmax(f1_score)]
        opt_f1_score = torch.max(f1_score)
        return opt_f1_score
