"""Dimensionality reduction and decomposition algorithms for feature processing.

This module provides implementations of dimensionality reduction techniques used
in anomaly detection models.

Classes:
    PCA: Principal Component Analysis for linear dimensionality reduction.
    SparseRandomProjection: Random projection using sparse random matrices.

Example:
    >>> from anomalib.models.components.dimensionality_reduction import PCA
    >>> # Create and fit PCA
    >>> pca = PCA(n_components=10)
    >>> features = torch.randn(100, 50)  # 100 samples, 50 features
    >>> reduced_features = pca.fit_transform(features)
    >>> # Use SparseRandomProjection
    >>> from anomalib.models.components.dimensionality_reduction import (
    ...     SparseRandomProjection
    ... )
    >>> projector = SparseRandomProjection(n_components=20)
    >>> projected_features = projector.fit_transform(features)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .pca import PCA
from .random_projection import SparseRandomProjection

__all__ = ["PCA", "SparseRandomProjection"]
