"""Process pool executor."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

from jsonargparse import Namespace
from rich import print
from rich.progress import Progress, TaskID

from anomalib.pipelines.jobs.base import Job
from anomalib.pipelines.runners.base import Runner

if TYPE_CHECKING:
    from concurrent.futures import Future

logger = logging.getLogger(__name__)


class ParallelExecutionError(Exception):
    """Pool execution error should be raised when one or more jobs fail in the pool."""


class ParallelRunner(Runner):
    """Run the job in parallel using a process pool."""

    def __init__(self, job: Job, n_jobs: int) -> None:
        super().__init__(job)
        self.n_jobs = n_jobs
        self.processes: dict[int, Future | None] = {}
        self.progress = Progress()
        self.task_id: TaskID
        self.results: list[dict] = []
        self.failures = False

    def run(self, args: Namespace) -> None:
        """Run the job in parallel."""
        self.task_id = self.progress.add_task(self.job.name, total=None)
        self.progress.start()
        self.processes = {i: None for i in range(self.n_jobs)}

        with ProcessPoolExecutor(max_workers=self.n_jobs, mp_context=multiprocessing.get_context("spawn")) as executor:
            for config in self.job.get_iterator(args):
                while None not in self.processes.values():
                    self._cleanup_processes()
                # get free index
                index = next(i for i, p in self.processes.items() if p is None)
                self.processes[index] = executor.submit(self.job.run, **config, task_id=index)
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
        gathered_result = self.job.collect(self.results)
        self.job.save(gathered_result)
        if self.failures:
            msg = f"[bold red]There were some errors with job {self.job.name}[/bold red]"
            print(msg)
            logger.error(msg)
            raise ParallelExecutionError(msg)
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
