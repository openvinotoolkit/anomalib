"""Benchmarking pipeline for evaluating anomaly detection models.

This module provides functionality for running benchmarking experiments that evaluate
and compare multiple anomaly detection models. The benchmarking pipeline supports
running experiments in parallel across multiple GPUs when available.

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

The pipeline handles setting up appropriate runners based on available hardware,
using parallel execution when multiple GPUs are available and serial execution
otherwise.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from anomalib.pipelines.components.base import Pipeline, Runner
from anomalib.pipelines.components.runners import ParallelRunner, SerialRunner

from .generator import BenchmarkJobGenerator


class Benchmark(Pipeline):
    """Benchmarking pipeline for evaluating anomaly detection models.

    This pipeline handles running benchmarking experiments that evaluate and compare
    multiple anomaly detection models. It supports both serial and parallel execution
    depending on available hardware.

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

    @staticmethod
    def _setup_runners(args: dict) -> list[Runner]:
        """Set up the appropriate runners for benchmark execution.

        This method configures either serial or parallel runners based on the
        specified accelerator(s) and available hardware. For CUDA devices, parallel
        execution is used when multiple GPUs are available.

        Args:
            args (dict): Dictionary containing configuration arguments. Must include
                an ``"accelerator"`` key specifying either a single accelerator or
                list of accelerators to use.

        Returns:
            list[Runner]: List of configured runner instances.

        Raises:
            ValueError: If an unsupported accelerator type is specified. Only
                ``"cpu"`` and ``"cuda"`` are supported.

        Example:
            >>> args = {"accelerator": "cuda"}
            >>> runners = Benchmark._setup_runners(args)
        """
        accelerators = args["accelerator"] if isinstance(args["accelerator"], list) else [args["accelerator"]]
        runners: list[Runner] = []
        for accelerator in accelerators:
            if accelerator not in {"cpu", "cuda"}:
                msg = f"Unsupported accelerator: {accelerator}"
                raise ValueError(msg)
            device_count = torch.cuda.device_count()
            if device_count <= 1 or accelerator == "cpu":
                runners.append(SerialRunner(BenchmarkJobGenerator(accelerator)))
            else:
                runners.append(ParallelRunner(BenchmarkJobGenerator(accelerator), n_jobs=device_count))
        return runners
