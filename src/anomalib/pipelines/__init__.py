"""Pipelines for end-to-end usecases."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .runners import ParallelRunner, SerialRunner

__all__ = ["ParallelRunner", "SerialRunner"]
