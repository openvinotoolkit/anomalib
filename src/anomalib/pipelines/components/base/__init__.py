"""Base classes for pipeline components in anomalib.

This module provides the core base classes used to build pipelines in anomalib:

- :class:`Job`: Base class for individual pipeline jobs
- :class:`JobGenerator`: Base class for generating pipeline jobs
- :class:`Runner`: Base class for executing pipeline jobs
- :class:`Pipeline`: Base class for creating complete pipelines

Example:
    >>> from anomalib.pipelines.components.base import Pipeline
    >>> from anomalib.pipelines.components.base import Runner
    >>> from anomalib.pipelines.components.base import Job, JobGenerator

    >>> # Create custom pipeline components
    >>> class MyJob(Job):
    ...     pass
    >>> class MyRunner(Runner):
    ...     pass
    >>> class MyPipeline(Pipeline):
    ...     pass

The base classes provide the foundation for building modular and extensible
pipelines for tasks like training, inference and benchmarking.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .job import Job, JobGenerator
from .pipeline import Pipeline
from .runner import Runner

__all__ = ["Job", "JobGenerator", "Runner", "Pipeline"]
