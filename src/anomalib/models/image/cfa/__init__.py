"""Implementation of the CFA (Coupled-hypersphere-based Feature Adaptation) model.

This module provides the CFA model for target-oriented anomaly localization. CFA
learns discriminative features by adapting them to coupled hyperspheres in the
feature space.

The model uses a teacher-student architecture where the teacher network extracts
features from normal samples to guide the student network in learning
anomaly-sensitive representations.

Paper: https://arxiv.org/abs/2206.04325

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models.image import Cfa
    >>> from anomalib.engine import Engine

    >>> # Initialize model and data
    >>> datamodule = MVTecAD()
    >>> model = Cfa()

    >>> # Train using the Engine
    >>> engine = Engine()
    >>> engine.fit(model=model, datamodule=datamodule)

    >>> # Get predictions
    >>> predictions = engine.predict(model=model, datamodule=datamodule)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import Cfa

__all__ = ["Cfa"]
