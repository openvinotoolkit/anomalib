"""Helpers for benchmarking and hyperparameter optimization."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .inference import get_openvino_throughput, get_torch_throughput

__all__ = ["get_openvino_throughput", "get_torch_throughput"]
