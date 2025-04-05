"""Label name enumeration class.

This module defines an enumeration class for labeling data in anomaly detection tasks.
The labels are represented as integers, where:

- ``UNKNOWN`` (-1): Represents samples with unknown/undefined labels
- ``NORMAL`` (0): Represents normal/good samples
- ``ABNORMAL`` (1): Represents anomalous/defective samples

Example:
    >>> from anomalib.data.utils.label import LabelName
    >>> label = LabelName.NORMAL
    >>> label.value
    0
    >>> label = LabelName.ABNORMAL
    >>> label.value
    1
    >>> label = LabelName.UNKNOWN
    >>> label.value
    -1
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class LabelName(int, Enum):
    """Enumeration class for labeling data in anomaly detection.

    This class inherits from both ``int`` and ``Enum`` to create an integer-based
    enumeration. This allows for easy comparison and conversion between label
    names and their corresponding integer values.

    Attributes:
        UNKNOWN (int): Label value -1, representing samples with unknown/undefined labels
        NORMAL (int): Label value 0, representing normal/good samples
        ABNORMAL (int): Label value 1, representing anomalous/defective samples
    """

    UNKNOWN = -1
    NORMAL = 0
    ABNORMAL = 1
