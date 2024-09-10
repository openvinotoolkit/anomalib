"""Anomalib Depth Datasets."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from .folder_3d import Folder3D
from .mvtec_3d import MVTec3D


class DepthDataFormat(str, Enum):
    """Supported Depth Dataset Types."""

    MVTEC_3D = "mvtec_3d"
    FOLDER_3D = "folder_3d"


__all__ = ["Folder3D", "MVTec3D"]
