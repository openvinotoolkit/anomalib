"""FastFlow Algorithm Implementation.

FastFlow is a fast flow-based anomaly detection model that uses normalizing flows
to model the distribution of features extracted from a pre-trained CNN backbone.
The model achieves competitive performance while maintaining fast inference times.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import Fastflow
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = Fastflow()
    >>> engine = Engine()

    >>> engine.fit(model, datamodule=datamodule)  # doctest: +SKIP
    >>> predictions = engine.predict(model, datamodule=datamodule)  # doctest: +SKIP

Paper:
    Title: FastFlow: Unsupervised Anomaly Detection and Localization via 2D
           Normalizing Flows
    URL: https://arxiv.org/abs/2111.07677

See Also:
    :class:`anomalib.models.image.fastflow.torch_model.FastflowModel`:
        PyTorch implementation of the FastFlow model architecture.
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import Fastflow
from .loss import FastflowLoss
from .torch_model import FastflowModel

__all__ = ["FastflowModel", "FastflowLoss", "Fastflow"]
