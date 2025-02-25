"""PatchCore: Towards Total Recall in Industrial Anomaly Detection.

PatchCore is an anomaly detection model that uses a memory bank of patch features
extracted from a pretrained CNN backbone. It stores representative patch features
from normal training images and detects anomalies by comparing test image patches
against this memory bank.

The model uses a nearest neighbor search to find the most similar patches in the
memory bank and computes anomaly scores based on these distances. It achieves
high performance while maintaining interpretability through localization maps.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import Patchcore
    >>> from anomalib.engine import Engine

    >>> # Initialize model and data
    >>> datamodule = MVTecAD()
    >>> model = Patchcore(
    ...     backbone="wide_resnet50_2",
    ...     layers=["layer2", "layer3"],
    ...     coreset_sampling_ratio=0.1
    ... )

    >>> # Train using the Engine
    >>> engine = Engine()
    >>> engine.fit(model=model, datamodule=datamodule)

    >>> # Get predictions
    >>> predictions = engine.predict(model=model, datamodule=datamodule)

Paper: https://arxiv.org/abs/2106.08265
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import Patchcore

__all__ = ["Patchcore"]
