# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RAD: Retrieval-based Anomaly Detection.

RAD is a training-free anomaly detection framework that stores anomaly-free
features in a multi-layer memory bank and detects anomalies through multi-level
retrieval with position-aware patch matching.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import Rad
    >>> from anomalib.engine import Engine

    >>> datamodule = MVTecAD()
    >>> model = Rad()

    >>> engine = Engine()
    >>> engine.fit(model=model, datamodule=datamodule)
    >>> predictions = engine.predict(model=model, datamodule=datamodule)
"""

from .lightning_model import Rad

__all__ = ["Rad"]
