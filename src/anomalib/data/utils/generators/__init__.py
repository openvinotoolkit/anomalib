"""Utilities to generate synthetic data."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .opensimplex import random_2d_simplex
from .perlin import random_2d_perlin

__all__ = ["random_2d_perlin", "random_2d_simplex"]
