"""Serial execution of pipeline jobs.

This module provides the :class:`SerialRunner` class for executing pipeline jobs
sequentially on a single device. It processes jobs one at a time in order.

Example:
    >>> from anomalib.pipelines.components.runners import SerialRunner
    >>> from anomalib.pipelines.components.base import JobGenerator
    >>> generator = JobGenerator()
    >>> runner = SerialRunner(generator)
    >>> results = runner.run({"param": "value"})

The serial runner handles:

- Sequential execution of jobs in order
- Progress tracking with progress bars
- Result collection and combination
- Error handling for failed jobs

This is useful when:

- Resources are limited to a single device
- Jobs need to be executed in a specific order
- Debugging pipeline execution
- Simple workflows that don't require parallelization

The runner implements the :class:`Runner` interface defined in
``anomalib.pipelines.components.base``.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from tqdm import tqdm

from anomalib.pipelines.components.base import JobGenerator, Runner
from anomalib.pipelines.types import GATHERED_RESULTS, PREV_STAGE_RESULT

logger = logging.getLogger(__name__)


class SerialExecutionError(Exception):
    """Error when running a job serially."""


class SerialRunner(Runner):
    """Serial executor for running jobs sequentially.

    This runner executes jobs one at a time in a sequential manner. It provides progress
    tracking and error handling while running jobs serially.

    Args:
        generator (JobGenerator): Generator that creates jobs to be executed.

    Example:
        Create a runner and execute jobs sequentially:

        >>> from anomalib.pipelines.components.runners import SerialRunner
        >>> from anomalib.pipelines.components.base import JobGenerator
        >>> generator = JobGenerator()
        >>> runner = SerialRunner(generator)
        >>> results = runner.run({"param": "value"})

    The runner handles:
        - Sequential execution of jobs
        - Progress tracking with progress bars
        - Result collection and combination
        - Error handling for failed jobs
    """

    def __init__(self, generator: JobGenerator) -> None:
        super().__init__(generator)

    def run(self, args: dict, prev_stage_results: PREV_STAGE_RESULT = None) -> GATHERED_RESULTS:
        """Execute jobs sequentially and gather results.

        This method runs each job one at a time, collecting results and handling any
        failures that occur during execution.

        Args:
            args (dict): Arguments specific to the job. For example, if there is a
                pipeline defined where one of the job generators is hyperparameter
                optimization, then the pipeline configuration file will look something
                like:

                .. code-block:: yaml

                    arg1:
                    arg2:
                    hpo:
                        param1:
                        param2:
                        ...

                In this case, ``args`` will receive a dictionary with all keys under
                ``hpo``.

            prev_stage_results (PREV_STAGE_RESULT, optional): Results from the previous
                pipeline stage. Used when the current stage depends on previous results.
                Defaults to None.

        Returns:
            GATHERED_RESULTS: Combined results from all executed jobs.

        Raises:
            SerialExecutionError: If any job fails during execution.
        """
        results = []
        failures = False
        logger.info(f"Running job {self.generator.job_class.name}")
        for job in tqdm(self.generator(args, prev_stage_results), desc=self.generator.job_class.name):
            try:
                results.append(job.run())
            except Exception:  # noqa: PERF203
                failures = True
                logger.exception("Error running job.")
        gathered_result = self.generator.job_class.collect(results)
        self.generator.job_class.save(gathered_result)
        if failures:
            msg = f"There were some errors with job {self.generator.job_class.name}"
            print(msg)
            logger.error(msg)
            raise SerialExecutionError(msg)
        logger.info(f"Job {self.generator.job_class.name} completed successfully.")
        return gathered_result
