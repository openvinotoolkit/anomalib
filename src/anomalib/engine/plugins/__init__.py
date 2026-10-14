# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Plugins for the Lightning Trainer used by Anomalib Engine."""

from .checkpoint_io import AnomalibCheckpointIO

__all__ = ["AnomalibCheckpointIO"]
