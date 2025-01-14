"""Base job class that defines the interface for pipeline jobs.

This module provides the base :class:`Job` class that all pipeline jobs inherit from. Jobs
are atomic units of work that can be executed independently, either serially or in
parallel.

Example:
    >>> from anomalib.pipelines.components.base import Job
    >>> class MyJob(Job):
    ...     name = "my_job"
    ...     def run(self, task_id=None):
    ...         # Implement job logic
    ...         pass
    ...     @staticmethod
    ...     def collect(results):
    ...         # Combine results from multiple runs
    ...         pass
    ...     @staticmethod
    ...     def save(results):
    ...         # Save final results
    ...         pass

The base job interface defines three key methods that subclasses must implement:

- :meth:`run`: Execute the core job logic
- :meth:`collect`: Gather and combine results from multiple job runs
- :meth:`save`: Save or export the final collected results

Jobs can be used as building blocks in pipelines for tasks like training,
inference, or benchmarking.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Generator

from anomalib.pipelines.types import GATHERED_RESULTS, PREV_STAGE_RESULT, RUN_RESULTS


class Job(ABC):
    """A job is an atomic unit of work that can be run in parallel with other jobs."""

    name: str

    @abstractmethod
    def run(self, task_id: int | None = None) -> RUN_RESULTS:
        """A job is a single unit of work that can be run in parallel with other jobs.

        ``task_id`` is optional and is only passed when the job is run in parallel.
        """

    @staticmethod
    @abstractmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Gather the results returned from run.

        This can be used to combine the results from multiple runs or to save/process individual job results.

        Args:
            results (list): List of results returned from run.

        Returns:
            (GATHERED_RESULTS): Collated results.
        """

    @staticmethod
    @abstractmethod
    def save(results: GATHERED_RESULTS) -> None:
        """Save the gathered results.

        This can be used to save the results in a file or a database.

        Args:
            results: The gathered result returned from gather_results.
        """


class JobGenerator(ABC):
    """Generate Job.

    The runners accept a generator that generates the jobs. The task of this class is to parse the config and return an
    iterator of specific jobs.
    """

    def __call__(
        self,
        args: dict | None = None,
        prev_stage_result: PREV_STAGE_RESULT = None,
    ) -> Generator[Job, None, None]:
        """Calls the ``generate_jobs`` method."""
        return self.generate_jobs(args, prev_stage_result)

    @abstractmethod
    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: PREV_STAGE_RESULT = None,
    ) -> Generator[Job, None, None]:
        """Return an iterator based on the arguments.

        This can be used to generate the configurations that will be passed to run.
        """

    @property
    @abstractmethod
    def job_class(self) -> type[Job]:
        """Return the job class that will be generated."""
