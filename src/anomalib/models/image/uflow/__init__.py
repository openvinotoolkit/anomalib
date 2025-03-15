"""U-Flow: A U-shaped Normalizing Flow for Anomaly Detection with Unsupervised Threshold.

This module implements the U-Flow model for anomaly detection as described in
Rudolph et al., 2022: U-Flow: A U-shaped Normalizing Flow for Anomaly Detection
with Unsupervised Threshold.

The model consists of:
- A U-shaped normalizing flow architecture for density estimation
- Unsupervised threshold estimation based on the learned density
- Anomaly detection by comparing likelihoods to the threshold

Example:
    >>> from anomalib.models.image import Uflow
    >>> from anomalib.engine import Engine
    >>> from anomalib.data import MVTecAD

    >>> datamodule = MVTecAD()
    >>> model = Uflow()
    >>> engine = Engine(model=model, datamodule=datamodule)

    >>> engine.fit()  # doctest: +SKIP
    >>> predictions = engine.predict()  # doctest: +SKIP

See Also:
    - :class:`anomalib.models.image.uflow.lightning_model.Uflow`:
        Lightning implementation of the model
    - :class:`anomalib.models.image.uflow.torch_model.UflowModel`:
        PyTorch implementation of the model architecture
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import Uflow

__all__ = ["Uflow"]
