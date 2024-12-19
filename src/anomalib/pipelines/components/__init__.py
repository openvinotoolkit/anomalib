"""Components for building and executing pipelines.

This module provides core components for constructing and running data processing
pipelines:

- :class:`Job`: Base class for defining pipeline jobs
- :class:`JobGenerator`: Creates job instances for pipeline stages
- :class:`Pipeline`: Manages execution flow between pipeline stages
- :class:`Runner`: Executes jobs serially or in parallel

Example:
    >>> from anomalib.pipelines.components import Pipeline, JobGenerator
    >>> generator = JobGenerator()
    >>> pipeline = Pipeline([generator])
    >>> pipeline.run({"param": "value"})

The components handle:
    - Job creation and configuration
    - Pipeline stage organization
    - Job execution and result gathering
    - Error handling and logging
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .base import Job, JobGenerator, Pipeline, Runner

__all__ = [
    "Job",
    "JobGenerator",
    "Pipeline",
    "Runner",
]
