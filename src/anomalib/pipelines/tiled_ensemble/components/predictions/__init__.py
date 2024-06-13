"""Module used for prediction storage and joining."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .prediction_data import (
    EnsemblePredictions,
)
from .prediction_merging import PredictionMergingMechanism

__all__ = [
    "EnsemblePredictions",
    "PredictionMergingMechanism",
]
