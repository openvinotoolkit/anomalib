"""Benchmark job generator."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Generator

from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.pipelines.components import JobGenerator
from anomalib.pipelines.components.utils import get_iterator_from_grid_dict
from anomalib.pipelines.types import PREV_STAGE_RESULT
from anomalib.utils.logging import hide_output

from .job import BenchmarkJob


def flatten_dict(d: dict | object, key: str = "", output: dict | None = None) -> dict:
    """Flatten a dictionary by dot-connecting the hierarchical keys."""
    # B006 Do not use mutable data structures for argument defaults.
    if output is None:
        output = {}
    if isinstance(d, dict):
        for k, v in d.items():
            flatten_dict(v, f"{key}.{k}" if key != "" else k, output)
    else:
        output[key] = d
    return output


class BenchmarkJobGenerator(JobGenerator):
    """Generate BenchmarkJob.

    Args:
        accelerator (str): The accelerator to use.
    """

    def __init__(self, accelerator: str) -> None:
        self.accelerator = accelerator

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return BenchmarkJob

    @hide_output
    def generate_jobs(
        self,
        args: dict,
        previous_stage_result: PREV_STAGE_RESULT,
    ) -> Generator[BenchmarkJob, None, None]:
        """Return iterator based on the arguments."""
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
