"""Implementation of Optimal F1 score based on TorchMetrics."""
import torch
from torchmetrics import Metric, PrecisionRecallCurve


class AdaptiveThreshold(Metric):
    """Optimal F1 Metric.

    Compute the optimal F1 score at the adaptive threshold, based on the F1 metric of the true labels and the
    predicted anomaly scores.
    """

    def __init__(self, default_value: float = 0.5, **kwargs):
        super().__init__(**kwargs)

        self.precision_recall_curve = PrecisionRecallCurve(num_classes=1, compute_on_step=False)
        self.add_state("value", default=torch.tensor(default_value), persistent=True)  # pylint: disable=not-callable
        self.value = torch.tensor(default_value)  # pylint: disable=not-callable

    # pylint: disable=arguments-differ
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # type: ignore
        """Update the precision-recall curve metric."""
        self.precision_recall_curve.update(preds, target)

    def compute(self) -> torch.Tensor:
        """Compute the threshold that yields the optimal F1 score.

        Compute the F1 scores while varying the threshold. Store the optimal
        threshold as attribute and return the maximum value of the F1 score.

        Returns:
            Value of the F1 score at the optimal threshold.
        """
        precision: torch.Tensor
        recall: torch.Tensor
        thresholds: torch.Tensor

        precision, recall, thresholds = self.precision_recall_curve.compute()
        f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
        if thresholds.dim() == 0:
            # special case where recall is 1.0 even for the highest threshold.
            # In this case 'thresholds' will be scalar.
            self.value = thresholds
        else:
            self.value = thresholds[torch.argmax(f1_score)]
        return self.value
