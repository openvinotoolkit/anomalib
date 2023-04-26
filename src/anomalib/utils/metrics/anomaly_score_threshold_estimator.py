"""Implementation of distribution estimator based AnomalyScoreThreshold."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.mixture import GaussianMixture
from torch import Tensor
from torchmetrics.utilities.data import dim_zero_cat

from anomalib.utils.metrics import AnomalyScoreThreshold

__all__ = ["GaussianMixtureThresholdEstimator"]


class BaseAnomalyScoreThresholdEstimator(AnomalyScoreThreshold, ABC):

    def __init__(self, default_value: float = 0.5, positive_rate: Optional[float] = None, **kwargs) -> None:
        super().__init__(default_value=default_value)
        assert positive_rate is None or 0. < positive_rate < 1., "Estimated positive rate should be in (0, 1) "
        self.positive_rate = positive_rate
        self.kwargs = kwargs

    def compute(self) -> Tensor:
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        warning_message = self._check_validity_warning_message(preds, target)
        if warning_message:
            warnings.warn(f"{warning_message} Falling back to Adaptive Threshold without density estimation.")
            return super().compute()

        thresholds = self._get_sorted_candidate_thresholds(preds, target)
        positive_cdf = self._run_estimator_cdf(preds[target == 1], thresholds)
        negative_cdf = self._run_estimator_cdf(preds[target == 0], thresholds)

        if self.positive_rate is not None:
            negative_alpha = (1 - self.positive_rate) / self.positive_rate
        else:
            negative_alpha = target.numel() / (target == 1).sum() - 1
        fp = (1. - negative_cdf) * negative_alpha
        tp = 1. - positive_cdf
        fn = positive_cdf
        f1_scores = (tp * 2.) / (tp * 2. + fp + fn)

        self.value = torch.tensor(thresholds[np.argmax(f1_scores)])  # pylint: disable=not-callable
        return self.value

    def _check_validity_warning_message(self, preds: Tensor, target: Tensor) -> Optional[str]:
        # pylint: disable=unused-argument
        return None

    def _get_sorted_candidate_thresholds(self, preds: Tensor, target: Tensor) -> Tensor:
        # pylint: disable=unused-argument
        return torch.sort(preds)[0].view(-1)

    @abstractmethod
    def _run_estimator_cdf(self, preds: Tensor, sorted_thresholds: Tensor) -> Tensor:
        pass


class GaussianMixtureThresholdEstimator(BaseAnomalyScoreThresholdEstimator):
    DEFAULT_POSITIVE_RATE = None
    DEFAULT_N_COMPONENTS = 1

    def __init__(self,
                 default_value: float = 0.5,
                 positive_rate: float | None = DEFAULT_POSITIVE_RATE,
                 n_components: int = DEFAULT_N_COMPONENTS,
                 **kwargs) -> None:
        super().__init__(default_value=default_value, positive_rate=positive_rate, **kwargs)
        self.n_components = n_components
        self.n_candidate_thresholds = 10 ** 5

    def _check_validity_warning_message(self, preds: Tensor, target: Tensor) -> Optional[str]:
        # pylint: disable=unused-argument
        min_samples = max(2, self.n_components)
        has_enough_data = (target == 0).sum() >= min_samples and (target == 1).sum() >= min_samples
        if not has_enough_data:
            return ("The validation set contains too few anomalous or normal images to conduct a "
                    "Gaussian Mixture estimator.")
        return None

    def _get_sorted_candidate_thresholds(self, preds: Tensor, target: Tensor) -> Tensor:
        # pylint: disable=unused-argument
        return torch.linspace(preds.min(), preds.max(), self.n_candidate_thresholds)

    def _run_estimator_cdf(self, preds: Tensor, sorted_thresholds: Tensor) -> Tensor:
        estimator = GaussianMixture(self.n_components, covariance_type='full', **self.kwargs)
        estimator.fit(preds.reshape(-1, 1).numpy())
        cdf = np.zeros(sorted_thresholds.shape)
        for weight, mean, var in zip(estimator.weights_, estimator.means_, estimator.covariances_):
            # TODO(yujie): var should be sqrt?
            mean = mean.flatten()[0]
            var = var.flatten()[0]
            cdf += norm.cdf(sorted_thresholds, loc=mean, scale=var ** 0.5) * weight
        return torch.from_numpy(cdf)
