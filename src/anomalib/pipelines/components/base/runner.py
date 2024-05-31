"""Base runner."""

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
