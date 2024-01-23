"""Implementation of AnomalyScoreThreshold based on TorchMetrics."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings

import numpy as np
import scipy.stats
import torch
from sklearn.mixture import GaussianMixture
from torch import Tensor
from torchmetrics import PrecisionRecallCurve
from torchmetrics.utilities.data import dim_zero_cat


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


class AnomalyScoreGaussianMixtureThreshold(AnomalyScoreThreshold):
    """Anomaly Score Threshold based on Gaussian Mixture in :meth:`sklearn.mixture.GaussianMixture`.

    This class is used to compute/store the threshold that determines the anomalous label
    given anomaly scores when the threshold method is ``gaussian_mixture``.

    The class conducts distribution estimation on normal and anomalous samples respectively
    with Gaussian Mixture Model, computes the adaptive threshold for optimal f1_score based
    on the data distributions, and finally stores the computed threshold value.
    """

    DEFAULT_ANOMALOUS_RATE = None
    DEFAULT_N_COMPONENTS = 2
    N_THRESHOLD_CANDIDATES = 10**5

    def __init__(
        self,
        default_value: float = 0.5,
        anomalous_rate: float | None = DEFAULT_ANOMALOUS_RATE,
        n_components: int = DEFAULT_N_COMPONENTS,
        **kwargs,
    ) -> None:
        """Initializes `AnomalyScoreGaussianMixtureThreshold` instance.

        Args:
            default_value (float): Default value of the threshold. [default: 0.5]
            anomalous_rate (float | None): The anticipated anomalous rate among all data samples, which can affect
                                          the final decision for optimal f1_score. This is useful when the stored
                                          predictions cannot reflect the actual anomalous rate in real cases (e.g.,
                                          anomalous samples are highly augmented) but we still want a practical
                                          threshold for real cases. If set to None, the anomalous rate will be
                                          automatically calculated based on the stored predictions. [default: None]
            n_components (int): The number of mixture components for Gaussian Mixture. [default: 2]
            **kwargs: Other arguments for initializing :meth:`sklearn.mixture.GaussianMixture`.
        """

        super().__init__(default_value=default_value, **kwargs)
        assert (
            anomalous_rate is None or 0.0 < anomalous_rate < 1.0
        ), "Anticipated anomalous rate should be set to None or should be in value range (0, 1)."
        self.anomalous_rate = anomalous_rate
        self.n_components = n_components
        self.kwargs = kwargs

    def compute(self) -> Tensor:
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        if not self._is_sufficient_dataset(target):
            warnings.warn(
                "The validation set contains too few anomalous or normal images to conduct a Gaussian "
                "Mixture estimation. Falling back to Adaptive Threshold without density estimation."
            )
            return super().compute()

        thresholds = self._get_sorted_candidate_thresholds(preds)
        positive_cdf = self._compute_estimated_cdf(preds[target == 1], thresholds)
        negative_cdf = self._compute_estimated_cdf(preds[target == 0], thresholds)
        anomalous_rate = self.anomalous_rate or float((target == 1).float().mean())
        f1_scores = self._compute_f1_scores_by_cdf(negative_cdf, positive_cdf, 1.0 - anomalous_rate, anomalous_rate)
        self.value = thresholds[torch.argmax(f1_scores)]
        return self.value

    def _is_sufficient_dataset(self, target: Tensor) -> bool:
        min_samples = max(2, self.n_components)
        return bool((target == 0).sum() >= min_samples and (target == 1).sum() >= min_samples)

    def _get_sorted_candidate_thresholds(self, preds: Tensor) -> Tensor:
        return torch.linspace(preds.min(), preds.max(), self.N_THRESHOLD_CANDIDATES)

    def _compute_estimated_cdf(self, preds: Tensor, sorted_thresholds: Tensor) -> Tensor:
        estimator = GaussianMixture(self.n_components, covariance_type="full", **self.kwargs)
        estimator.fit(preds.reshape(-1, 1).numpy())
        cdf = np.zeros(sorted_thresholds.shape)
        for weight, mean, covar in zip(estimator.weights_, estimator.means_, estimator.covariances_):
            mean = mean.flatten()[0]
            covar = covar.flatten()[0]
            cdf += scipy.stats.norm.cdf(sorted_thresholds, loc=mean, scale=covar**0.5) * weight
        return torch.from_numpy(cdf)

    def _compute_f1_scores_by_cdf(
        self, negative_cdf: Tensor, positive_cdf: Tensor, negative_weight: float, positive_weight: float
    ) -> Tensor:
        fp = (1.0 - negative_cdf) * negative_weight
        tp = (1.0 - positive_cdf) * positive_weight
        fn = positive_cdf * positive_weight
        f1_scores = (tp * 2.0) / (tp * 2.0 + fp + fn)
        return f1_scores
