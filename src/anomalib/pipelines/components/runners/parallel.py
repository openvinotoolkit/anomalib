"""Process pool executor."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

from anomalib.pipelines.components.base import JobGenerator, Runner
from anomalib.pipelines.types import GATHERED_RESULTS, PREV_STAGE_RESULT

if TYPE_CHECKING:
    from concurrent.futures import Future

logger = logging.getLogger(__name__)


class ParallelExecutionError(Exception):
    """Pool execution error should be raised when one or more jobs fail in the pool."""


class ParallelRunner(Runner):
    """Run the job in parallel using a process pool.

    It creates a pool of processes and submits the jobs to the pool.
    This is useful when you have fixed resources that you want to re-use.
    Once a process is done, it is replaced with a new job.

    Args:
        generator (JobGenerator): The generator that generates the jobs.
        n_jobs (int): The number of jobs to run in parallel.

    Example:
        Creating a pool with the size of the number of available GPUs and submitting jobs to the pool.
        >>> ParallelRunner(generator, n_jobs=torch.cuda.device_count())
        Each time a job is submitted to the pool, an additional parameter `task_id` will be passed to `job.run` method.
        The job can then use this `task_id` to assign a particular device to train on.
        >>> def run(self, arg1: int, arg2: nn.Module, task_id: int) -> None:
        >>>     device = torch.device(f"cuda:{task_id}")
        >>>     model = arg2.to(device)
        >>>     ...

    """

    def __init__(self, generator: JobGenerator, n_jobs: int) -> None:
        super().__init__(generator)
        self.n_jobs = n_jobs
        self.processes: dict[int, Future | None] = {}
        self.results: list[dict] = []
        self.failures = False

    def run(self, args: dict, prev_stage_results: PREV_STAGE_RESULT = None) -> GATHERED_RESULTS:
        """Run the job in parallel."""
        self.processes = dict.fromkeys(range(self.n_jobs))

        with ProcessPoolExecutor(max_workers=self.n_jobs, mp_context=multiprocessing.get_context("spawn")) as executor:
            for job in self.generator(args, prev_stage_results):
                while None not in self.processes.values():
                    self._await_cleanup_processes()
                # get free index
                index = next(i for i, p in self.processes.items() if p is None)
                self.processes[index] = executor.submit(job.run, task_id=index)
            self._await_cleanup_processes(blocking=True)

        gathered_result = self.generator.job_class.collect(self.results)
        self.generator.job_class.save(gathered_result)
        if self.failures:
            msg = f"There were some errors with job {self.generator.job_class.name}"
            print(msg)
            logger.error(msg)
            raise ParallelExecutionError(msg)
        logger.info(f"Job {self.generator.job_class.name} completed successfully.")
        return gathered_result

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
