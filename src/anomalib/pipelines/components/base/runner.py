"""Base runner class for executing pipeline jobs.

This module provides the abstract base class for runners that execute pipeline jobs.
Runners handle the mechanics of job execution, whether serial or parallel.

Example:
    >>> from anomalib.pipelines.components.base import Runner
    >>> from anomalib.pipelines.components.base import JobGenerator
    >>> class MyRunner(Runner):
    ...     def run(self, args: dict, prev_stage_results=None):
    ...         # Implement runner logic
    ...         pass

    >>> # Create and use runner
    >>> generator = JobGenerator()
    >>> runner = MyRunner(generator)
    >>> results = runner.run({"param": "value"})

The base runner interface defines the core :meth:`run` method that subclasses must
implement to execute jobs. Runners work with job generators to create and execute
pipeline jobs.

Runners can implement different execution strategies like:

- Serial execution of jobs one after another
- Parallel execution across multiple processes
- Distributed execution across machines
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from anomalib.pipelines.types import GATHERED_RESULTS, PREV_STAGE_RESULT

from .job import JobGenerator


class Runner(ABC):
    """Base runner.

    Args:
        generator (JobGenerator): Job generator.
    """

    def __init__(self, generator: JobGenerator) -> None:
        self.generator = generator

    @abstractmethod
    def run(self, args: dict, prev_stage_results: PREV_STAGE_RESULT = None) -> GATHERED_RESULTS:
        """Run the pipeline.

        Args:
            args (dict): Arguments specific to the job. For example, if there is a pipeline defined where one of the job
                generators is hyperparameter optimization, then the pipeline configuration file will look something like
                ```yaml
                arg1:
                arg2:
                hpo:
                    param1:
                    param2:
                    ...
                ```
                In this case, the `args` will receive a dictionary with all keys under `hpo`.

            prev_stage_results (PREV_STAGE_RESULT, optional): Previous stage results. This is useful when the current
                stage depends on the results of the previous stage. Defaults to None.

        Returns:
            GATHERED_RESULTS: Gathered results from all the jobs.
        """
