"""Job from which all the jobs inherit from."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from jsonargparse import ArgumentParser, Namespace


class ConfigParser(ABC):
    """Base class for config parsers."""

    def __init__(self, name: str) -> None:
        self.logger = logging.getLogger(name)

    @abstractmethod
    def add_arguments(self, parser: ArgumentParser) -> None:
        """Add arguments to the parser."""

    @abstractmethod
    def config_iterator(self, args: Namespace) -> Iterator:
        """Return iterator based on the arguments."""


class Job(ABC):
    """A job is an atomic unit of work that can be run in parallel with other jobs.

    Args:
        config_parser (BaseConfigParser): A parser that will be used to parse the arguments.
        name (str): Name of the job. This is used by the progress bar and logging.
    """

    def __init__(self, config_parser: ConfigParser, name: str) -> None:
        self.config_parser = config_parser
        self.name = name
        self.logger = logging.getLogger(self.name)

    @abstractmethod
    def run(self, *args, task_id: int | None = None) -> dict[str, Any]:
        """A job is a single unit of work that can be run in parallel with other jobs.

        ``task_id`` is optional and is only passed when the job is run in parallel.
        """

    @abstractmethod
    def gather_results(self, results: list[Any]) -> Any:  # noqa: ANN401
        """Gather the results returned from run.

        This can be used to combine the results from multiple runs or to save/process individual job results.

        Args:
            results (list): List of results returned from run.
        """

    @abstractmethod
    def tabulate_results(self, results: Any) -> dict[str, Any]:  # noqa: ANN401
        """Return the results as a dict so that it can be shown as a table in the terminal and also saved as a csv file.

        Should return a dictionary like
            {
                "column1": [1,2,3,4,5],
                "column2": [6,7,8,9,10],
            }

        """

    @abstractmethod
    def save_gathered_result(self, result: Any) -> None:  # noqa: ANN401
        """Save the gathered results.

        This can be used to save the results in a file or a database.

        Args:
            result: The gathered result returned from gather_results.
        """

    def config_iterator(self, args: Namespace) -> Iterator:
        """Convenience method to call the config_parser's config_iterator."""
        return self.config_parser.config_iterator(args)

    def add_arguments(self, parser: ArgumentParser) -> None:
        """Convenience method to call the config_parser's _add_arguments."""
        self.config_parser.add_arguments(parser)
