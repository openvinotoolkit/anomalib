"""Unit Tests - Datamodule Configurations."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from anomalib import TaskType


@pytest.fixture(params=[TaskType.CLASSIFICATION, TaskType.DETECTION, TaskType.SEGMENTATION])
def task_type(request: type[pytest.FixtureRequest]) -> str:
    """Create and return a task type."""
    return request.param
