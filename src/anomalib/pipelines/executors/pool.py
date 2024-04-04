"""Process pool executor."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import torch
from jsonargparse import ArgumentParser, Namespace
from rich import print
from rich.progress import Progress, TaskID

from anomalib.pipelines.executors.serial import SerialExecutor
from anomalib.pipelines.jobs.base import BaseJob

logger = logging.getLogger(__name__)


class PoolExecutionError(Exception):
    """Pool execution error should be raised when one or more jobs fail in the pool."""


class PoolExecutor(SerialExecutor):
    def __init__(self, job: BaseJob):
        super().__init__(job)
        self.processes = {}
        self.progress = Progress()
        self.task_id: TaskID
        self.results = []
        self.failures = False

    def run(self, args: Namespace) -> None:
        """Run the job in parallel."""
        self.task_id = self.progress.add_task(self.job.name, total=None)
        self.progress.start()
        self.processes = {i: None for i in range(args.n_jobs)}

        with ProcessPoolExecutor(max_workers=args.n_jobs, mp_context=multiprocessing.get_context("spawn")) as executor:
            for config in self.job.config_iterator(args):
                while None not in self.processes.values():
                    self._cleanup_processes()
                # get free index
                index = next(i for i, p in self.processes.items() if p is None)
                self.processes[index] = executor.submit(self.job.run, config, task_id=index)
            while None not in self.processes.values():
                self._cleanup_processes()
            for process in self.processes.values():
                if process is not None:
                    try:
                        self.results.append(process.result())
                    except Exception:
                        logger.exception("An exception occurred while getting the process result.")
                        self.failures = True

        self.progress.update(self.task_id, completed=1, total=1)
        self.progress.stop()
        gathered_result = self.job.gather_results(self.results)
        self.job.save_gathered_result(gathered_result)
        self._print_tabular_results(gathered_result)
        if self.failures:
            msg = f"[bold red]There were some errors with job {self.job.name}[/bold red]"
            print(msg)
            logger.error(msg)
            raise PoolExecutionError(msg)
        logger.info(f"Job {self.job.name} completed successfully.")

    def _cleanup_processes(self) -> None:
        """Wait for any one process to finish."""
        for index, process in self.processes.items():
            if process is not None and process.done():
                try:
                    self.results.append(process.result())
                except Exception:
                    logger.exception("An exception occurred while getting the process result.")
                    self.failures = True
                self.processes[index] = None
                self.progress.update(self.task_id, advance=1)

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add n_jobs to the parser."""
        parser.add_argument(
            "--n_jobs",
            type=int,
            default=torch.cuda.device_count(),
            help="Number of jobs to run in parallel.",
        )
        self.job.add_arguments(parser)
