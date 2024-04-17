"""Executor for running a job serially."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from jsonargparse import Namespace
from rich import print
from rich.progress import track

from anomalib.pipelines.jobs.base import Job
from anomalib.pipelines.runners.base import Runner

logger = logging.getLogger(__name__)


class SerialExecutionError(Exception):
    """Error when running a job serially."""


class SerialRunner(Runner):
    """Serial executor for running a single job at a time."""

    def __init__(self, job: Job) -> None:
        super().__init__(job)

    def run(self, args: Namespace) -> None:
        """Run the job."""
        results = []
        failures = False
        logger.info(f"Running job {self.job.name}")
        for config in track(self.job.get_iterator(args), description=self.job.name):
            try:
                results.append(self.job.run(**config))
            except Exception:  # noqa: PERF203
                failures = True
                logger.exception(f"Error running job with config {config}")
        gathered_result = self.job.on_collect(results)
        self.job.on_save(gathered_result)
        if failures:
            msg = f"[bold red]There were some errors with job {self.job.name}[/bold red]"
            print(msg)
            logger.error(msg)
            raise SerialExecutionError(msg)
        logger.info(f"Job {self.job.name} completed successfully.")
