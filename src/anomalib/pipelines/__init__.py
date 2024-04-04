"""Pipelines for end-to-end usecases."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .executors import PoolExecutor, SerialExecutor
from .pipeline import Pipeline

__all__ = ["Pipeline", "PoolExecutor", "SerialExecutor"]
