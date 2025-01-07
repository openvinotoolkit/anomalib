"""Sampling methods for anomaly detection models.

This module provides sampling techniques used in anomaly detection models to
select representative samples from datasets.

Classes:
    KCenterGreedy: K-center greedy sampling algorithm that selects diverse and
        representative samples.

Example:
    >>> import torch
    >>> from anomalib.models.components.sampling import KCenterGreedy
    >>> # Create sampler
    >>> sampler = KCenterGreedy()
    >>> # Sample from feature embeddings
    >>> features = torch.randn(100, 512)  # 100 samples with 512 dimensions
    >>> selected_idx = sampler.select_coreset(features, n=10)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .k_center_greedy import KCenterGreedy

__all__ = ["KCenterGreedy"]
