"""Student-Teacher Feature Pyramid Matching Model for anomaly detection.

This module implements the STFPM model for anomaly detection as described in
Wang et al., 2021: Student-Teacher Feature Pyramid Matching for Unsupervised
Anomaly Detection.

The model consists of:
- A pre-trained teacher network that extracts multi-scale features
- A student network that learns to match the teacher's feature representations
- Feature pyramid matching between student and teacher features
- Anomaly detection based on feature discrepancy

Example:
    >>> from anomalib.models.image import Stfpm
    >>> from anomalib.engine import Engine
    >>> from anomalib.data import MVTecAD

    >>> datamodule = MVTecAD()
    >>> model = Stfpm()
    >>> engine = Engine(model=model, datamodule=datamodule)

    >>> engine.fit()  # doctest: +SKIP
    >>> predictions = engine.predict()  # doctest: +SKIP

See Also:
    - :class:`anomalib.models.image.stfpm.lightning_model.Stfpm`:
        Lightning implementation of the model
    - :class:`anomalib.models.image.stfpm.torch_model.StfpmModel`:
        PyTorch implementation of the model architecture
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import Stfpm

__all__ = ["Stfpm"]
