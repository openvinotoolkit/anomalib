"""Base runner."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from anomalib.pipelines.types import GATHERED_RESULTS, PREV_STAGE_RESULT

from .job import JobGenerator


class Runner(ABC):
    """Base runner."""

    def __init__(self, generator: JobGenerator) -> None:
        self.generator = generator

    @abstractmethod
    def run(self, args: dict, prev_stage_results: PREV_STAGE_RESULT = None) -> GATHERED_RESULTS:
        """Run the pipeline."""
