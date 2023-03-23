"""Unit Tests - Base Datamodules."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .datamodule import _TestAnomalibDataModule
from .depth import _TestAnomalibDepthDatamodule
from .image import _TestAnomalibImageDatamodule
from .video import _TestAnomalibVideoDatamodule

__all__ = [
    "_TestAnomalibDataModule",
    "_TestAnomalibDepthDatamodule",
    "_TestAnomalibImageDatamodule",
    "_TestAnomalibVideoDatamodule",
]
