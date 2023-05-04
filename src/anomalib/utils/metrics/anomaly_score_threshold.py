"""Implementation of AnomalyScoreThreshold based on TorchMetrics."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from enum import Enum

import torch
from torch import Tensor
from torchmetrics import PrecisionRecallCurve


class AnomalyScoreThreshold(PrecisionRecallCurve):
    """Anomaly Score Threshold.

    This class computes/stores the threshold that determines the anomalous label
    given anomaly scores. If the threshold method is ``manual``, the class only
    stores the manual threshold values.

    If the threshold method is ``adaptive``, the class initially computes the
    adaptive threshold to find the optimal f1_score and stores the computed
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
            warnings.warn(
                "The validation set does not contain any anomalous images. As a result, the adaptive threshold will "
                "take the value of the highest anomaly score observed in the normal validation images, which may lead "
                "to poor predictions. For a more reliable adaptive threshold computation, please add some anomalous "
                "images to the validation set."
            )
        precision, recall, thresholds = super().compute()
        f1_score = (2 * precision * recall) / (precision + recall + 1e-10)
        if thresholds.dim() == 0:
            # special case where recall is 1.0 even for the highest threshold.
            # In this case 'thresholds' will be scalar.
            self.value = thresholds
        else:
            self.value = thresholds[torch.argmax(f1_score)]
        return self.value


class MeanVarianceMode(str, Enum):
    """Supported Mean Variance Combination Type"""

    ADD = "add"
    MINUS = "minus"


class AnomalyScoreMeanVarThreshold(AnomalyScoreThreshold):
    """Anomaly Score Threshold.

    This class computes/stores the threshold that determines the anomalous label
    given anomaly scores. The computation is based on the mean and variance.

    """

    STD_DEV_RATIO = 0.1
    MEANVAR_MODE = MeanVarianceMode.ADD

    def __init__(self, default_value: float = 0.5, **kwargs) -> None:
        self.meanvar_mode = kwargs.get("meanvar_mode", self.MEANVAR_MODE)
        self.std_dev_ratio = kwargs.get("std_dev_ratio", self.STD_DEV_RATIO)
        super().__init__(default_value=default_value, **kwargs)

    def compute(self) -> Tensor:
        """Compute the threshold.

        Compute the threshold based on the mean and variance of the predictions

        Returns:
            Value of threshold.
        """

        if not any(1 in batch for batch in self.target):
            warnings.warn(
                "The validation set does not contain any anomalous images. As a result, the adaptive threshold will "
                "take the value of the highest anomaly score observed in the normal validation images, which may lead "
                "to poor predictions. For a more reliable adaptive threshold computation, please add some anomalous "
                "images to the validation set."
            )
        mean_pred = torch.mean(torch.cat(self.preds, dim=0))
        std_pred = torch.std(torch.cat(self.preds, dim=0))
        if self.meanvar_mode == MeanVarianceMode.MINUS:
            self.value = mean_pred - self.std_dev_ratio * std_pred
        else:
            self.value = mean_pred + self.std_dev_ratio * std_pred
        return self.value
