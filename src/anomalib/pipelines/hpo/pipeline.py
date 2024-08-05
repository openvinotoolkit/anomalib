"""Hyperparameter optimization."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib.pipelines.components.base import Pipeline, Runner
from anomalib.pipelines.components.runners import SerialRunner

from .comet_job_generator import CometHPOJobGenerator
from .wandb_job_generator import WandbHPOJobGenerator


class HPO(Pipeline):
    """Hyperparameter optimization pipeline."""

    def _setup_runners(self, args: dict) -> list[Runner]:
        """Setup runners for the HPO."""
        backend = args.get("backend", "comet")
        job = CometHPOJobGenerator() if backend == "comet" else WandbHPOJobGenerator()
        return [SerialRunner(job)]
