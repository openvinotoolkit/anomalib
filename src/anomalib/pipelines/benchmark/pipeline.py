"""Benchmarking."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from anomalib.pipelines.components.base import Pipeline, Runner
from anomalib.pipelines.components.runners import ParallelRunner, SerialRunner

from .generator import BenchmarkJobGenerator


class Benchmark(Pipeline):
    """Benchmarking pipeline."""

    @staticmethod
    def _setup_runners(args: dict) -> list[Runner]:
        """Setup the runners for the pipeline."""
        accelerators = args["accelerator"] if isinstance(args["accelerator"], list) else [args["accelerator"]]
        runners: list[Runner] = []
        for accelerator in accelerators:
            if accelerator not in {"cpu", "cuda"}:
                msg = f"Unsupported accelerator: {accelerator}"
                raise ValueError(msg)
            device_count = torch.cuda.device_count()
            if device_count <= 1:
                runners.append(SerialRunner(BenchmarkJobGenerator(accelerator)))
            else:
                runners.append(ParallelRunner(BenchmarkJobGenerator(accelerator), n_jobs=device_count))
        return runners
