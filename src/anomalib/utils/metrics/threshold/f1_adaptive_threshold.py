"""Implementation of F1AdaptiveThreshold based on TorchMetrics."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging

import torch
from torch import Tensor
from torchmetrics import PrecisionRecallCurve

from .base import BaseThreshold

logger = logging.getLogger(__name__)


class F1AdaptiveThreshold(PrecisionRecallCurve, BaseThreshold):
    """Anomaly Score Threshold.

    This class computes/stores the threshold that determines the anomalous label
    given anomaly scores.
    The class initially computes the adaptive threshold to find the optimal f1_score and stores the computed
    adaptive threshold value.
    """

    def __init__(self, default_value: float = 0.5, **kwargs) -> None:
        super().__init__(num_classes=1, **kwargs)

        self.add_state("value", default=torch.tensor(default_value), persistent=True)  # pylint: disable=not-callable
        self.value = torch.tensor(default_value)  # pylint: disable=not-callable

    def compute(self) -> Tensor:
        """Compute the threshold that yields the optimal F1 score.

        Compute the F1 scores while varying the threshold. Store the optimal
        threshold as attribute and return the maximum value of the F1 score.

        Returns:
            Value of the F1 score at the optimal threshold.
        """
        precision: Tensor
        recall: Tensor
        thresholds: Tensor

        if not any(1 in batch for batch in self.target):
            msg = (
                "The validation set does not contain any anomalous images. As a result, the adaptive threshold will "
                "take the value of the highest anomaly score observed in the normal validation images, which may lead "
                "to poor predictions. For a more reliable adaptive threshold computation, please add some anomalous "
                "images to the validation set."
            )
            logging.warn(msg)

        precision, recall, thresholds = super().compute()
        f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
        if thresholds.dim() == 0:
            # special case where recall is 1.0 even for the highest threshold.
            # In this case 'thresholds' will be scalar.
            self.value = thresholds
        else:
            self.value = thresholds[torch.argmax(f1_score)]
        return self.value
