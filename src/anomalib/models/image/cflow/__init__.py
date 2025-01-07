"""Real-Time Unsupervised Anomaly Detection via Conditional Normalizing Flows.

This module provides the implementation of CFLOW model for anomaly detection.
CFLOW uses conditional normalizing flows to model the distribution of normal
samples in the feature space.

Example:
    >>> from anomalib.models.image.cflow import Cflow
    >>> # Initialize the model
    >>> model = Cflow(
    ...     backbone="resnet18",
    ...     flow_steps=8,
    ...     hidden_ratio=1.0,
    ...     coupling_blocks=4,
    ...     clamp_alpha=1.9,
    ...     permute_soft=False
    ... )
    >>> # Forward pass
    >>> x = torch.randn(32, 3, 256, 256)
    >>> predictions = model(x)

Paper: https://arxiv.org/abs/2107.12571
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import Cflow

__all__ = ["Cflow"]
