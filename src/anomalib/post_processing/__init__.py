"""Anomalib post-processing module."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .base import PostProcessor
from .one_class import OneClassPostProcessor

__all__ = ["OneClassPostProcessor", "PostProcessor"]
