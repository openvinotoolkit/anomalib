"""Executor for running a job serially."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from jsonargparse import ArgumentParser, Namespace
from rich import print
from rich.console import Console
from rich.progress import track
from rich.table import Table

from anomalib.pipelines.jobs.base import BaseJob

logger = logging.getLogger(__name__)


class SerialExecutionError(Exception):
    """Error when running a job serially."""


class SerialExecutor:
    """Serial executor for running a single job."""

    def __init__(self, job: BaseJob) -> None:
        self.job = job

    def run(self, args: Namespace) -> None:
        """Run the job."""
        results = []
        failures = False
        logger.info(f"Running job {self.job.name}")
        for config in track(self.job.config_iterator(args), description=self.job.name):
            try:
                results.append(self.job.run(config))
            except Exception as exception:
                failures = True
                logger.exception(f"Error running job with config {config}: {exception}")
        gathered_result = self.job.gather_results(results)
        self.job.save_gathered_result(gathered_result)
        self._print_tabular_results(gathered_result)
        if failures:
            msg = f"[bold red]There were some errors with job {self.job.name}[/bold red]"
            print(msg)
            logger.error(msg)
            raise SerialExecutionError(msg)
        logger.info(f"Job {self.job.name} completed successfully.")

    def _print_tabular_results(self, gathered_result: Any | None = None) -> None:
        """Print the tabular results."""
        if gathered_result is not None:
            console = Console()
            table = Table(title=f"{self.job.name} Results", show_header=True, header_style="bold magenta")
            _results = self.job.tabulate_results(gathered_result)
            for column in _results.keys():
                table.add_column(column)
            for row in zip(*_results.values(), strict=False):
                table.add_row(*[str(value) for value in row])
            console.print(table)

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Convenience method to add arguments from the job to the parser.

        Can also be used to add additional arguments to the parser.
        """
        self.job.add_arguments(parser)

    def process_args(self, args: Namespace) -> Namespace:
        """Process the arguments."""
        return self.job.config_parser.preprocess_args(args)
