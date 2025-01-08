"""Utilities to generate synthetic data.

This module provides utilities for generating synthetic data for anomaly detection.
The utilities include:

- Perlin noise generation: Functions for creating Perlin noise patterns
- Anomaly generation: Classes for generating synthetic anomalies

Example:
    >>> from anomalib.data.utils.generators import generate_perlin_noise
    >>> # Generate 256x256 Perlin noise
    >>> noise = generate_perlin_noise(256, 256)
    >>> print(noise.shape)
    torch.Size([256, 256])

    >>> from anomalib.data.utils.generators import PerlinAnomalyGenerator
    >>> # Create anomaly generator
    >>> generator = PerlinAnomalyGenerator()
    >>> # Generate anomaly mask
    >>> mask = generator.generate(256, 256)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .perlin import PerlinAnomalyGenerator, generate_perlin_noise

__all__ = ["PerlinAnomalyGenerator", "generate_perlin_noise"]
