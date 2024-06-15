"""Tiled ensemble pipelines."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .test_pipeline import TestTiledEnsemble
from .train_pipeline import TrainTiledEnsemble

__all__ = [
    "TrainTiledEnsemble",
    "TestTiledEnsemble",
]
