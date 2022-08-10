"""Algorithms for decomposition and dimensionality reduction."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .pca import PCA
from .random_projection import SparseRandomProjection

__all__ = ["PCA", "SparseRandomProjection"]
