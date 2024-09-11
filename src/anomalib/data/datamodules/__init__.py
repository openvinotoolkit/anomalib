"""Anomalib Data Modules."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .depth import Folder3D, MVTec3D
from .image import BTech, Folder, Kolektor, MVTec, Visa
from .video import Avenue, ShanghaiTech, UCSDped

__all__ = [
    "Folder3D",
    "MVTec3D",
    "BTech",
    "Folder",
    "Kolektor",
    "MVTec",
    "Visa",
    "Avenue",
    "ShanghaiTech",
    "UCSDped",
]
