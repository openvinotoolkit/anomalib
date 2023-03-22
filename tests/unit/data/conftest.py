"""Unit Tests - Datamodules Configurations."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from anomalib.data import TaskType
import pytest


@pytest.fixture(params=[TaskType.CLASSIFICATION, TaskType.DETECTION, TaskType.SEGMENTATION])
def task_type(request) -> str:
    """Create and return a task type."""
    return request.param
