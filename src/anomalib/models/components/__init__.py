"""Components used within the anomaly detection models.

This module provides various components that are used across different anomaly
detection models in the library.

Components:
    Base Components:
        - ``AnomalibModule``: Base module for all anomaly detection models
        - ``BufferListMixin``: Mixin for managing lists of buffers
        - ``DynamicBufferMixin``: Mixin for dynamic buffer management
        - ``MemoryBankMixin``: Mixin for memory bank functionality

    Dimensionality Reduction:
        - ``PCA``: Principal Component Analysis
        - ``SparseRandomProjection``: Random projection with sparse matrices

    Feature Extraction:
        - ``TimmFeatureExtractor``: Feature extractor using timm models
        - ``TorchFXFeatureExtractor``: Feature extractor using TorchFX

    Image Processing:
        - ``GaussianBlur2d``: 2D Gaussian blur filter

    Sampling:
        - ``KCenterGreedy``: K-center greedy sampling algorithm

    Statistical Methods:
        - ``GaussianKDE``: Gaussian kernel density estimation
        - ``MultiVariateGaussian``: Multivariate Gaussian distribution

Example:
    >>> from anomalib.models.components import GaussianKDE
    >>> kde = GaussianKDE()
    >>> # Use components in anomaly detection models
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .base import AnomalibModule, BufferListMixin, DynamicBufferMixin, MemoryBankMixin
from .dimensionality_reduction import PCA, SparseRandomProjection
from .feature_extractors import TimmFeatureExtractor, TorchFXFeatureExtractor
from .filters import GaussianBlur2d
from .sampling import KCenterGreedy
from .stats import GaussianKDE, MultiVariateGaussian

__all__ = [
    "AnomalibModule",
    "BufferListMixin",
    "DynamicBufferMixin",
    "MemoryBankMixin",
    "GaussianKDE",
    "GaussianBlur2d",
    "KCenterGreedy",
    "MultiVariateGaussian",
    "PCA",
    "SparseRandomProjection",
    "TimmFeatureExtractor",
    "TorchFXFeatureExtractor",
]
