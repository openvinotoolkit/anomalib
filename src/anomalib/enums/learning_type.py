"""Learning type enum."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class LearningType(str, Enum):
    """Learning type defining how the model learns from the dataset samples."""

    ONE_CLASS = "one_class"
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
