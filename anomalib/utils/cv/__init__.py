"""Anomalib operators."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .connected_components import connected_components_cpu, connected_components_gpu

__all__ = ["connected_components_cpu", "connected_components_gpu"]
