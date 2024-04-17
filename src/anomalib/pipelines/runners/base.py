"""Base runner."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from jsonargparse import Namespace

from anomalib.pipelines.jobs.base import Job


class Runner(ABC):
    """Base runner."""

    def __init__(self, job: Job) -> None:
        self.job = job

    @abstractmethod
    def run(self, args: Namespace) -> None:
        """Run the pipeline."""
