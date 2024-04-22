"""Executor for running a job serially."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from jsonargparse import Namespace
from rich import print
from rich.progress import track

from anomalib.pipelines.components.base import JobGenerator, Runner

logger = logging.getLogger(__name__)


class SerialExecutionError(Exception):
    """Error when running a job serially."""


class SerialRunner(Runner):
    """Serial executor for running a single job at a time."""

    def __init__(self, generator: JobGenerator) -> None:
        super().__init__(generator)

    def run(self, args: Namespace) -> None:
        """Run the job."""
        results = []
        failures = False
        logger.info(f"Running job {self.generator.job_class.name}")
        for job in track(self.generator(args), description=self.generator.job_class.name):
            try:
                results.append(job.run())
            except Exception:  # noqa: PERF203
                failures = True
                logger.exception("Error running job.")
        gathered_result = self.generator.job_class.collect(results)
        self.generator.job_class.save(gathered_result)
        if failures:
            msg = f"[bold red]There were some errors with job {self.generator.job_class.name}[/bold red]"
            print(msg)
            logger.error(msg)
            raise SerialExecutionError(msg)
        logger.info(f"Job {self.generator.job_class.name} completed successfully.")
