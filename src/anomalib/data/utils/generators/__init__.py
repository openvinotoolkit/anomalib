"""Utilities to generate synthetic data."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .perlin import PerlinAnomalyGenerator, generate_perlin_noise

__all__ = ["PerlinAnomalyGenerator", "generate_perlin_noise"]
