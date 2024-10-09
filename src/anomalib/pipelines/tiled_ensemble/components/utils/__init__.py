"""Tiled ensemble utils and helper functions."""

from enum import Enum

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


class NormalizationStage(str, Enum):
    """Enum signaling at which stage the normalization is done.

    In case of tile, tiles are normalized for each tile position separately.
    In case of image, normalization is done at the end when images are joined back together.
    In case of none, output is not normalized.
    """

    TILE = "tile"
    IMAGE = "image"
    NONE = "none"


class ThresholdStage(str, Enum):
    """Enum signaling at which stage the thresholding is applied.

    In case of tile, thresholding is applied for each tile location separately.
    In case of image, thresholding is applied at the end when images are joined back together.
    """

    TILE = "tile"
    IMAGE = "image"


class PredictData(Enum):
    """Enum indicating which data to use in prediction job."""

    VAL = "val"
    TEST = "test"


__all__ = [
    "NormalizationStage",
    "ThresholdStage",
    "PredictData",
]
