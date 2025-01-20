"""Pipelines for end-to-end anomaly detection use cases.

This module provides high-level pipeline implementations for common anomaly detection
workflows:

- :class:`Benchmark`: Pipeline for benchmarking model performance across datasets

The pipelines handle:
    - Configuration and setup
    - Data loading and preprocessing
    - Model training and evaluation
    - Result collection and analysis
    - Logging and visualization

Example:
    >>> from anomalib.pipelines import Benchmark
    >>> benchmark = Benchmark(config_path="config.yaml")
    >>> results = benchmark.run()

The pipelines leverage components from :mod:`anomalib.pipelines.components` for:
    - Job management and execution
    - Parameter grid search
    - Result gathering
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .benchmark import Benchmark

__all__ = ["Benchmark"]
