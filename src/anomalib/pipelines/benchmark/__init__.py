"""Benchmarking pipeline for anomaly detection models.

This module provides functionality for benchmarking anomaly detection models in
anomalib. The benchmarking pipeline allows evaluating and comparing multiple models
across different datasets and metrics.

Example:
    >>> from anomalib.pipelines import Benchmark
    >>> from anomalib.data import MVTecAD
    >>> from anomalib.models import Padim, Patchcore

    >>> # Initialize benchmark with models and datasets
    >>> benchmark = Benchmark(
    ...     models=[Padim(), Patchcore()],
    ...     datasets=[MVTecAD(category="bottle"), MVTecAD(category="cable")]
    ... )

    >>> # Run benchmark
    >>> results = benchmark.run()
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .pipeline import Benchmark

__all__ = ["Benchmark"]
