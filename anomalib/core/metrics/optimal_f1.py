"""
Implementation of Optimal F1 score based on TorchMetrics.
"""
import torch
from torch import Tensor
from torchmetrics import Metric, PrecisionRecallCurve


class OptimalF1(Metric):
    """
    Compute the optimal F1 score at the adaptive threshold, based on the F1 metric of the
    true labels and the predicted anomaly scores.
    """

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)

        self.precision_recall_curve = PrecisionRecallCurve(num_classes=num_classes, compute_on_step=False)

        self.threshold: Tensor

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore  # pylint: disable=arguments-differ
        self.precision_recall_curve.update(preds, target)

    def compute(self) -> Tensor:
        """
        Compute the F1 scores while varying the threshold. Store the optimal
        threshold as attribute and return the maximum value of the F1 score.

        Returns:
            Value of the F1 score at the optimal threshold.
        """
        precision: Tensor
        recall: Tensor
        thresholds: Tensor

        precision, recall, thresholds = self.precision_recall_curve.compute()
        f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
        self.threshold = thresholds[torch.argmax(f1_score)]
        optimal_f1_score = torch.max(f1_score)
        return optimal_f1_score
