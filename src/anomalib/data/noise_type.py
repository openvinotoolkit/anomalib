"""Noise type enum."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class NoiseType(str, Enum):
    """Supported Noise Types"""

    PERLIN_2D = "perlin_2d"
    SIMPLEX_2D = "simplex_2d"
