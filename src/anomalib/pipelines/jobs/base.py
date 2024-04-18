"""Job from which all the jobs inherit from."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator

from jsonargparse import ArgumentParser, Namespace

from anomalib.pipelines.types import GATHERED_RESULTS, RUN_RESULTS


class Job(ABC):
    """A job is an atomic unit of work that can be run in parallel with other jobs."""

    name: str

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.name)

    @abstractmethod
    def run(self, task_id: int | None = None, **kwargs) -> RUN_RESULTS:
        """A job is a single unit of work that can be run in parallel with other jobs.

        ``task_id`` is optional and is only passed when the job is run in parallel.
        """

    @abstractmethod
    def collect(self, results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Gather the results returned from run.

        This can be used to combine the results from multiple runs or to save/process individual job results.

        Args:
            results (list): List of results returned from run.
        """

    @abstractmethod
    def save(self, results: GATHERED_RESULTS) -> None:
        """Save the gathered results.

        This can be used to save the results in a file or a database.

        Args:
            results: The gathered result returned from gather_results.
        """

    @staticmethod
    @abstractmethod
    def add_arguments(parser: ArgumentParser) -> None:
        """Add arguments to the parser.

        This can be used to add arguments that are specific to the job.
        """

    @staticmethod
    @abstractmethod
    def get_iterator(args: Namespace | None = None) -> Iterator:
        """Return an iterator based on the arguments.

        This can be used to generate the configurations that will be passed to run.
        """
