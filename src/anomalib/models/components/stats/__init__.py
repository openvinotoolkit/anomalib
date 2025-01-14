"""Statistical functions for anomaly detection models.

This module provides statistical methods used in anomaly detection models for
density estimation and probability modeling.

Classes:
    GaussianKDE: Gaussian kernel density estimation for non-parametric density
        estimation.
    MultiVariateGaussian: Multivariate Gaussian distribution for parametric
        density modeling.

Example:
    >>> import torch
    >>> from anomalib.models.components.stats import GaussianKDE
    >>> # Create density estimator
    >>> kde = GaussianKDE()
    >>> # Fit and evaluate density
    >>> features = torch.randn(100, 10)  # 100 samples, 10 dimensions
    >>> kde.fit(features)
    >>> density = kde.predict(features)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .kde import GaussianKDE
from .multi_variate_gaussian import MultiVariateGaussian

__all__ = ["GaussianKDE", "MultiVariateGaussian"]
