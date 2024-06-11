"""Label name enum class."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class LabelName(int, Enum):
    """Name of label."""

    NORMAL = 0
    ABNORMAL = 1
