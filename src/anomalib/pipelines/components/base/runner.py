"""Base runner."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from jsonargparse import Namespace

from .job import JobGenerator


class Runner(ABC):
    """Base runner."""

    def __init__(self, generator: JobGenerator) -> None:
        self.generator = generator

    @abstractmethod
    def run(self, args: Namespace) -> None:
        """Run the pipeline."""
