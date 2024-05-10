"""Job from which all the jobs inherit from."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Generator

from anomalib.pipelines.types import GATHERED_RESULTS, RUN_RESULTS


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
    """Generate Job."""

    def __call__(self, args: dict | None = None) -> Generator[Job, None, None]:
        """Calls the ``generate_jobs`` method."""
        return self.generate_jobs(args)

    @abstractmethod
    def generate_jobs(self, args: dict | None = None) -> Generator[Job, None, None]:
        """Return an iterator based on the arguments.

        This can be used to generate the configurations that will be passed to run.
        """

    @property
    @abstractmethod
    def job_class(self) -> type[Job]:
        """Return the job class that will be generated."""
