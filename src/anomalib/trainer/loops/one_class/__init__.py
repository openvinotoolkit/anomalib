"""Training loops for one-class anomaly detection."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .fit import AnomalibFitLoop as FitLoop
from .predict import AnomalibPredictionLoop as PredictionLoop
from .test import AnomalibTestLoop as TestLoop
from .validate import AnomalibValidationLoop as ValidationLoop

__all__ = ["FitLoop", "PredictionLoop", "TestLoop", "ValidationLoop"]
