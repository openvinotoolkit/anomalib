"""Clustering algorithm implementations using PyTorch.

This module provides clustering algorithms implemented in PyTorch for anomaly
detection tasks.

Classes:
    GaussianMixture: Gaussian Mixture Model for density estimation and clustering.
    KMeans: K-Means clustering algorithm.

Example:
    >>> from anomalib.models.components.cluster import GaussianMixture, KMeans
    >>> # Create and fit a GMM
    >>> gmm = GaussianMixture(n_components=3)
    >>> features = torch.randn(100, 10)  # Example features
    >>> gmm.fit(features)
    >>> # Create and fit KMeans
    >>> kmeans = KMeans(n_clusters=5)
    >>> kmeans.fit(features)
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .gmm import GaussianMixture
from .kmeans import KMeans

__all__ = ["GaussianMixture", "KMeans"]
