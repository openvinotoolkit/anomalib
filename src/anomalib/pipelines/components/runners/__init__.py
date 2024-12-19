"""Runners for executing pipeline jobs.

This module provides runner implementations for executing pipeline jobs in different
ways:

- :class:`SerialRunner`: Executes jobs sequentially on a single device
- :class:`ParallelRunner`: Executes jobs in parallel across multiple devices

Example:
    >>> from anomalib.pipelines.components.runners import SerialRunner
    >>> from anomalib.pipelines.components.base import JobGenerator
    >>> generator = JobGenerator()
    >>> runner = SerialRunner(generator)
    >>> results = runner.run({"param": "value"})

The runners handle the mechanics of job execution while working with job generators
to create and execute pipeline jobs. They implement the :class:`Runner` interface
defined in ``anomalib.pipelines.components.base``.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .parallel import ParallelRunner
from .serial import SerialRunner

__all__ = ["SerialRunner", "ParallelRunner"]
