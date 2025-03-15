"""Reverse Distillation Model for anomaly detection.

This module implements the Reverse Distillation model for anomaly detection as described in
the paper "Reverse Distillation: A New Training Strategy for Feature Reconstruction
Networks in Anomaly Detection" (Deng et al., 2022).

The model consists of:
- A pre-trained encoder (e.g. ResNet) that extracts multi-scale features
- A bottleneck layer that compresses features into a compact representation
- A decoder that reconstructs features back to the original feature space
- A scoring mechanism based on reconstruction error

Example:
    >>> from anomalib.models import ReverseDistillation
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.engine import Engine

    >>> # Initialize model and data
    >>> datamodule = MVTecAD()
    >>> model = ReverseDistillation()

    >>> # Train using the Engine
    >>> engine = Engine()
    >>> engine.fit(model=model, datamodule=datamodule)

    >>> # Get predictions
    >>> predictions = engine.predict(model=model, datamodule=datamodule)

See Also:
    - :class:`anomalib.models.image.reverse_distillation.lightning_model.ReverseDistillation`:
        Lightning implementation of the model
    - :class:`anomalib.models.image.reverse_distillation.torch_model.ReverseDistillationModel`:
        PyTorch implementation of the model architecture
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import ReverseDistillation

__all__ = ["ReverseDistillation"]
