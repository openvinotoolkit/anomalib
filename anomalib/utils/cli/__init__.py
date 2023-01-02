"""Anomalib CLI."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .benchmark import distribute
from .cli import AnomalibCLI

__all__ = ["AnomalibCLI", "distribute"]
