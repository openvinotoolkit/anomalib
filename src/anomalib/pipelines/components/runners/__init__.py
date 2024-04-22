"""Executor for running a single job."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .parallel import ParallelRunner
from .serial import SerialRunner

__all__ = ["SerialRunner", "ParallelRunner"]
