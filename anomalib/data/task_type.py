"""Task type enum."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class TaskType(str, Enum):
    """Task type used when generating predictions on the dataset."""

    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
