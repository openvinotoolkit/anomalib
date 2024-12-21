"""WinCLIP Model for anomaly detection.

This module implements anomaly detection using the WinCLIP model, which leverages
CLIP embeddings and a sliding window approach to detect anomalies in images.

Example:
    >>> from anomalib.models import WinClip
    >>> from anomalib.data import Visa
    >>> from anomalib.engine import Engine

    >>> # Initialize model and data
    >>> datamodule = Visa()
    >>> model = WinClip()

    >>> # Validate using the Engine
    >>> engine = Engine()
    >>> engine.validate(model=model, datamodule=datamodule)

    >>> # Get predictions
    >>> predictions = engine.predict(model=model, datamodule=datamodule)

See Also:
    - :class:`WinClip`: Main model class for WinCLIP-based anomaly detection
    - :class:`WinClipModel`: PyTorch implementation of the WinCLIP model
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import WinClip
from .torch_model import WinClipModel

__all__ = ["WinClip", "WinClipModel"]
