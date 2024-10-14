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
            if accelerator == "cpu":
                runners.append(SerialRunner(BenchmarkJobGenerator("cpu")))
            elif accelerator == "cuda":
                runners.append(ParallelRunner(BenchmarkJobGenerator("cuda"), n_jobs=torch.cuda.device_count()))
            else:
                msg = f"Unsupported accelerator: {accelerator}"
                raise ValueError(msg)
        return runners
