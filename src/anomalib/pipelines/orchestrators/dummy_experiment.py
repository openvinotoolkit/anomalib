"""Dummy Experiment."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from jsonargparse import ArgumentParser, Namespace

from anomalib.pipelines.jobs.dummy_experiment import CompareJob, ExtractJob, FitJob
from anomalib.pipelines.orchestrators.base import Orchestrator
from anomalib.pipelines.runners import ParallelRunner, SerialRunner
from anomalib.pipelines.runners.base import Runner


class Experiment(Orchestrator):
    """Dummy experiment."""

    def _setup_runners(self, args: Namespace) -> list[Runner]:
        """Setup the runners for the pipeline."""
        del args  # there is no job specific arguments
        return [
            ParallelRunner(ExtractJob(), n_jobs=1),
            SerialRunner(FitJob()),
            ParallelRunner(CompareJob(), n_jobs=1),
        ]

    def get_parser(self, parser: ArgumentParser | None = None) -> ArgumentParser:
        """Add arguments to the parser."""
        parser = super().get_parser(parser)
        ExtractJob.add_arguments(parser)
        FitJob.add_arguments(parser)
        return parser
