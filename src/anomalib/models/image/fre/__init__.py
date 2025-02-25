"""Feature Reconstruction Error (FRE) Algorithm Implementation.

FRE is an anomaly detection model that uses feature reconstruction error to detect
anomalies. The model extracts features from a pre-trained CNN backbone and learns
to reconstruct them using an autoencoder. Anomalies are detected by measuring the
reconstruction error.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import Fre
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = Fre()
    >>> engine = Engine()

    >>> engine.fit(model, datamodule=datamodule)  # doctest: +SKIP
    >>> predictions = engine.predict(model, datamodule=datamodule)  # doctest: +SKIP

See Also:
    :class:`anomalib.models.image.fre.lightning_model.Fre`:
        PyTorch Lightning implementation of the FRE model.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import Fre

__all__ = ["Fre"]
