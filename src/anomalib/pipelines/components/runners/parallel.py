"""Parallel execution of pipeline jobs using process pools.

This module provides the :class:`ParallelRunner` class for executing pipeline jobs in
parallel across multiple processes. It uses Python's :class:`ProcessPoolExecutor` to
manage a pool of worker processes.

Example:
    >>> from anomalib.pipelines.components.runners import ParallelRunner
    >>> from anomalib.pipelines.components.base import JobGenerator
    >>> generator = JobGenerator()
    >>> runner = ParallelRunner(generator, n_jobs=4)
    >>> results = runner.run({"param": "value"})

The parallel runner handles:

- Creating and managing a pool of worker processes
- Distributing jobs across available workers
- Collecting and combining results from parallel executions
- Error handling for failed jobs

The number of parallel jobs can be configured based on available compute resources
like CPU cores or GPUs.
"""

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
    """Run jobs in parallel using a process pool.

    This runner executes jobs concurrently using a pool of worker processes. It manages
    process creation, job distribution, and result collection.

    Args:
        generator (JobGenerator): Generator that creates jobs to be executed.
        n_jobs (int): Number of parallel processes to use.

    Example:
        Create a pool with size matching available GPUs and submit jobs:

        >>> from anomalib.pipelines.components.runners import ParallelRunner
        >>> from anomalib.pipelines.components.base import JobGenerator
        >>> import torch
        >>> generator = JobGenerator()
        >>> runner = ParallelRunner(generator, n_jobs=torch.cuda.device_count())
        >>> results = runner.run({"param": "value"})

    Notes:
        When a job is submitted to the pool, a ``task_id`` parameter is passed to the
        job's ``run()`` method. Jobs can use this ID to manage device assignment:

        .. code-block:: python

            def run(self, arg1: int, arg2: nn.Module, task_id: int) -> None:
                device = torch.device(f"cuda:{task_id}")
                model = arg2.to(device)
                # ... rest of job logic

    The runner handles:
        - Creating and managing worker processes
        - Distributing jobs to available workers
        - Collecting and combining results
        - Error handling for failed jobs
        - Resource cleanup
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
