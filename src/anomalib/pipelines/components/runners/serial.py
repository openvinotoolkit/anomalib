"""Executor for running a job serially."""

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
    """Serial executor for running a single job at a time."""

    def __init__(self, generator: JobGenerator) -> None:
        super().__init__(generator)

    def run(self, args: dict, prev_stage_results: PREV_STAGE_RESULT = None) -> GATHERED_RESULTS:
        """Run the job."""
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
