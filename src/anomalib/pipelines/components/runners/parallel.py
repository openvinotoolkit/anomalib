"""Process pool executor."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

from rich import print
from rich.progress import Progress, TaskID

from anomalib.pipelines.components.base import JobGenerator, Runner

if TYPE_CHECKING:
    from concurrent.futures import Future

logger = logging.getLogger(__name__)


class ParallelExecutionError(Exception):
    """Pool execution error should be raised when one or more jobs fail in the pool."""


class ParallelRunner(Runner):
    """Run the job in parallel using a process pool."""

    def __init__(self, generator: JobGenerator, n_jobs: int) -> None:
        super().__init__(generator)
        self.n_jobs = n_jobs
        self.processes: dict[int, Future | None] = {}
        self.progress = Progress()
        self.task_id: TaskID
        self.results: list[dict] = []
        self.failures = False

    def run(self, args: dict) -> None:
        """Run the job in parallel."""
        self.task_id = self.progress.add_task(self.generator.job_class.name, total=None)
        self.progress.start()
        self.processes = {i: None for i in range(self.n_jobs)}

        with ProcessPoolExecutor(max_workers=self.n_jobs, mp_context=multiprocessing.get_context("spawn")) as executor:
            for job in self.generator.generate_jobs(args):
                while None not in self.processes.values():
                    self._await_cleanup_processes()
                # get free index
                index = next(i for i, p in self.processes.items() if p is None)
                self.processes[index] = executor.submit(job.run, task_id=index)
            self._await_cleanup_processes(blocking=True)

        self.progress.update(self.task_id, completed=1, total=1)
        self.progress.stop()
        gathered_result = self.generator.job_class.collect(self.results)
        self.generator.job_class.save(gathered_result)
        if self.failures:
            msg = f"[bold red]There were some errors with job {self.generator.job_class.name}[/bold red]"
            print(msg)
            logger.error(msg)
            raise ParallelExecutionError(msg)
        logger.info(f"Job {self.generator.job_class.name} completed successfully.")

    def _await_cleanup_processes(self, blocking: bool = False) -> None:
        """Wait for any one process to finish.

        Args:
            blocking (bool): If True, wait for all processes to finish.
        """
        for index, process in self.processes.items():
            if process is not None and ((process.done() and not blocking) or blocking):
                try:
                    self.results.append(process.result())
                except Exception:
                    logger.exception("An exception occurred while getting the process result.")
                    self.failures = True
                self.processes[index] = None
                self.progress.update(self.task_id, advance=1)
