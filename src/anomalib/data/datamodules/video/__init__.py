"""Anomalib Video Data Modules.

This module contains data modules for loading and processing video datasets for
anomaly detection. The following data modules are available:

- ``Avenue``: CUHK Avenue Dataset for abnormal event detection
- ``ShanghaiTech``: ShanghaiTech Campus Dataset for anomaly detection
- ``UCSDped``: UCSD Pedestrian Dataset for anomaly detection

Example:
    Load the Avenue dataset::

        >>> from anomalib.data import Avenue
        >>> datamodule = Avenue(
        ...     root="./datasets/avenue",
        ...     clip_length_in_frames=2,
        ...     frames_between_clips=1
        ... )
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from .avenue import Avenue
from .shanghaitech import ShanghaiTech
from .ucsd_ped import UCSDped


class VideoDataFormat(str, Enum):
    """Supported Video Dataset Types.

    The following dataset formats are supported:

    - ``UCSDPED``: UCSD Pedestrian Dataset
    - ``AVENUE``: CUHK Avenue Dataset
    - ``SHANGHAITECH``: ShanghaiTech Campus Dataset
    """

    UCSDPED = "ucsdped"
    AVENUE = "avenue"
    SHANGHAITECH = "shanghaitech"


__all__ = ["Avenue", "ShanghaiTech", "UCSDped"]
