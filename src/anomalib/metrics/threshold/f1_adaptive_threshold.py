"""Implementation of F1AdaptiveThreshold based on TorchMetrics."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import torch

from anomalib.metrics.precision_recall_curve import BinaryPrecisionRecallCurve

from .base import Threshold

logger = logging.getLogger(__name__)


class F1AdaptiveThreshold(BinaryPrecisionRecallCurve, Threshold):
    """Anomaly Score Threshold.

    This class computes/stores the threshold that determines the anomalous label
    given anomaly scores. It initially computes the adaptive threshold to find
    the optimal f1_score and stores the computed adaptive threshold value.

    Args:
        default_value: Default value of the threshold.
            Defaults to ``0.5``.

    Examples:
        To find the best threshold that maximizes the F1 score, we could run the
        following:

        >>> from anomalib.metrics import F1AdaptiveThreshold
        >>> import torch
        ...
        >>> labels = torch.tensor([0, 0, 0, 1, 1])
        >>> preds = torch.tensor([2.3, 1.6, 2.6, 7.9, 3.3])
        ...
        >>> adaptive_threshold = F1AdaptiveThreshold(default_value=0.5)
        >>> threshold = adaptive_threshold(preds, labels)
        >>> threshold
        tensor(3.3000)
    """

    def __init__(self, default_value: float = 0.5, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_state("value", default=torch.tensor(default_value), persistent=True)
        self.value = torch.tensor(default_value)

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

        if not any(1 in batch for batch in self.target):
            msg = (
                "The validation set does not contain any anomalous images. As a result, the adaptive threshold will "
                "take the value of the highest anomaly score observed in the normal validation images, which may lead "
                "to poor predictions. For a more reliable adaptive threshold computation, please add some anomalous "
                "images to the validation set."
            )
            logging.warning(msg)

        precision, recall, thresholds = super().compute()
        f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
        if thresholds.dim() == 0:
            # special case where recall is 1.0 even for the highest threshold.
            # In this case 'thresholds' will be scalar.
            self.value = thresholds
        else:
            self.value = thresholds[torch.argmax(f1_score)]
        return self.value

    def __repr__(self) -> str:
        """Return threshold value within the string representation.

        Returns:
            str: String representation of the class.
        """
        return f"{super().__repr__()} (value={self.value:.2f})"
