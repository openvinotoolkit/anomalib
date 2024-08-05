"""HPO sweep job generator."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator

from anomalib.pipelines.components import JobGenerator
from anomalib.pipelines.components.base.job import Job
from anomalib.pipelines.types import PREV_STAGE_RESULT
from anomalib.utils.logging import hide_output

from .wandb_job import WandbHPOJob


class WandbHPOJobGenerator(JobGenerator):
    """Generate Wandb based HPO job."""

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return WandbHPOJob

    @hide_output
    def generate_jobs(
        self,
        config: dict,
        prev_stage_result: PREV_STAGE_RESULT,
    ) -> Generator[Job, None, None]:
        """Generate a single job that uses ``wandb.agent`` internally."""
        del prev_stage_result  # Not needed for this job
        yield WandbHPOJob(
            project=config["project"],
            entity=config["entity"],
            sweep_configuration=config["sweep_configuration"],
            count=config["count"],
        )
