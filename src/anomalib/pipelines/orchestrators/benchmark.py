"""Benchmarking."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from jsonargparse import ArgumentParser, Namespace

from anomalib.pipelines.jobs.benchmark import BenchmarkJob
from anomalib.pipelines.orchestrators.base import Orchestrator
from anomalib.pipelines.runners.base import Runner
from anomalib.pipelines.runners.parallel import ParallelRunner
from anomalib.pipelines.runners.serial import SerialRunner


class Benchmark(Orchestrator):
    """Benchmarking orchestrator."""

    def _setup_runners(self, args: Namespace) -> list[Runner]:
        """Setup the runners for the pipeline."""
        # TODO(ashwinvaidya17): refactor this to remove duplicate code
        if isinstance(args.accelerator, list):
            # TODO(ashwinvaidya17): wrap runners in a ParallelRunner
            runners: list[Runner] = []
            for accelerator in args.accelerator:
                if accelerator == "cpu":
                    runners.append(SerialRunner(BenchmarkJob(accelerator="cpu")))
                elif accelerator == "cuda":
                    runners.append(ParallelRunner(BenchmarkJob(accelerator="cuda"), n_jobs=torch.cuda.device_count()))
                else:
                    msg = f"Unsupported accelerator: {accelerator}"
                    raise ValueError(msg)
        elif args.accelerator == "cpu":
            runners = [SerialRunner(BenchmarkJob(accelerator="cpu"))]
        elif args.accelerator == "cuda":
            runners = [ParallelRunner(BenchmarkJob(accelerator="cuda"), n_jobs=torch.cuda.device_count())]
        else:
            msg = f"Unsupported accelerator: {args.accelerator}"
            raise ValueError(msg)
        return runners

    def get_parser(self, parser: ArgumentParser | None = None) -> ArgumentParser:
        """Add arguments to the parser."""
        parser = super().get_parser(parser)
        parser.add_argument(
            "--accelerator",
            type=str | list[str],
            default="cuda",
            help="Hardware to run the benchmark on.",
        )
        BenchmarkJob.add_arguments(parser)
        return parser
