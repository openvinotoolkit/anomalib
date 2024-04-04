"""Executor for running a single job."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .pool import PoolExecutor
from .serial import SerialExecutor

__all__ = ["SerialExecutor", "PoolExecutor"]
