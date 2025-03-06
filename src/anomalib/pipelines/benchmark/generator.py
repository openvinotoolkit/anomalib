"""Benchmark job generator for running model benchmarking experiments.

This module provides functionality for generating benchmark jobs that evaluate model
performance. It generates jobs based on provided configurations for models,
datasets and other parameters.

Example:
    >>> from anomalib.pipelines.benchmark.generator import BenchmarkJobGenerator
    >>> generator = BenchmarkJobGenerator(accelerator="gpu")
    >>> args = {
    ...     "seed": 42,
    ...     "model": {"class_path": "Padim"},
    ...     "data": {"class_path": "MVTecAD", "init_args": {"category": "bottle"}}
    ... }
    >>> jobs = list(generator.generate_jobs(args, None))

The generator creates :class:`BenchmarkJob` instances that can be executed to run
benchmarking experiments with specified models and datasets.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator

from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.pipelines.components import JobGenerator
from anomalib.pipelines.components.utils import get_iterator_from_grid_dict
from anomalib.pipelines.types import PREV_STAGE_RESULT
from anomalib.utils.config import flatten_dict
from anomalib.utils.logging import hide_output

from .job import BenchmarkJob


class BenchmarkJobGenerator(JobGenerator):
    """Generate benchmark jobs for evaluating model performance.

    This class generates benchmark jobs based on provided configurations for models,
    datasets and other parameters. Each job evaluates a specific model-dataset
    combination.

    Args:
        accelerator (str): Type of accelerator to use for running the jobs (e.g.
            ``"cpu"``, ``"gpu"``).

    Example:
        >>> from anomalib.pipelines.benchmark.generator import BenchmarkJobGenerator
        >>> generator = BenchmarkJobGenerator(accelerator="gpu")
        >>> args = {
        ...     "seed": 42,
        ...     "model": {"class_path": "Padim"},
        ...     "data": {"class_path": "MVTecAD", "init_args": {"category": "bottle"}}
        ... }
        >>> jobs = list(generator.generate_jobs(args, None))
    """

    def __init__(self, accelerator: str) -> None:
        self.accelerator = accelerator

    @property
    def job_class(self) -> type:
        """Get the job class used by this generator.

        Returns:
            type: The :class:`BenchmarkJob` class.
        """
        return BenchmarkJob

    @hide_output
    def generate_jobs(
        self,
        args: dict,
        previous_stage_result: PREV_STAGE_RESULT,
    ) -> Generator[BenchmarkJob, None, None]:
        """Generate benchmark jobs from the provided arguments.

        Args:
            args (dict): Dictionary containing job configuration including model,
                dataset and other parameters.
            previous_stage_result (PREV_STAGE_RESULT): Results from previous pipeline
                stage (unused).

        Yields:
            Generator[BenchmarkJob, None, None]: Generator yielding benchmark job
                instances.

        Example:
            >>> generator = BenchmarkJobGenerator(accelerator="cpu")
            >>> args = {
            ...     "seed": 42,
            ...     "model": {"class_path": "Padim"},
            ...     "data": {"class_path": "MVTecAD"}
            ... }
            >>> jobs = list(generator.generate_jobs(args, None))
        """
        del previous_stage_result  # Not needed for this job
        for _container in get_iterator_from_grid_dict(args):
            # Pass experimental configs as a flatten dictionary to the job runner.
            flat_cfg = flatten_dict(_container)
            yield BenchmarkJob(
                accelerator=self.accelerator,
                seed=_container["seed"],
                model=get_model(_container["model"]),
                datamodule=get_datamodule(_container["data"]),
                flat_cfg=flat_cfg,
            )
