"""Loops for default strategy."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from .fit import AnomalibFitLoop, AnomalibTrainingEpochLoop
from .predict import AnomalibPredictionEpochLoop, AnomalibPredictionLoop
from .test import AnomalibTestEpochLoop, AnomalibTestLoop
from .validate import AnomalibValidationEpochLoop, AnomalibValidationLoop

__all__ = [
    "AnomalibFitLoop",
    "AnomalibTrainingEpochLoop",
    "AnomalibValidationLoop",
    "AnomalibValidationEpochLoop",
    "AnomalibTestLoop",
    "AnomalibTestEpochLoop",
    "AnomalibPredictionEpochLoop",
    "AnomalibPredictionLoop",
]
