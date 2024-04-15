"""Anomalib Video Datasets."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from .avenue import Avenue
from .folder_video import FolderVideo
from .shanghaitech import ShanghaiTech
from .ucsd_ped import UCSDped


class VideoDataFormat(str, Enum):
    """Supported Video Dataset Types."""

    UCSDPED = "ucsdped"
    AVENUE = "avenue"
    SHANGHAITECH = "shanghaitech"
    FOLDERVIDEO = "foldervideo"


__all__ = ["Avenue", "ShanghaiTech", "UCSDped", "FolderVideo"]
