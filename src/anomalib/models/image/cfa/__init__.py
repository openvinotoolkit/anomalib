"""Implementation of the CFA (Coupled-hypersphere-based Feature Adaptation) model.

This module provides the CFA model for target-oriented anomaly localization. CFA
learns discriminative features by adapting them to coupled hyperspheres in the
feature space.

The model uses a teacher-student architecture where the teacher network extracts
features from normal samples to guide the student network in learning
anomaly-sensitive representations.

Paper: https://arxiv.org/abs/2206.04325

Example:
    >>> from anomalib.models.image import Cfa
    >>> # Initialize the model
    >>> model = Cfa()
    >>> # Train on normal samples
    >>> model.fit(normal_samples)
    >>> # Get anomaly predictions
    >>> predictions = model.predict(test_samples)
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import Cfa

__all__ = ["Cfa"]
