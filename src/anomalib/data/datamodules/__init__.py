"""Anomalib Data Modules."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .depth import Folder3D, MVTec3D
from .image import VAD, BTech, Datumaro, Folder, Kolektor, MVTec, MVTecAD, Visa
from .video import Avenue, ShanghaiTech, UCSDped

__all__ = [
    "Folder3D",
    "MVTec3D",
    "BTech",
    "Datumaro",
    "Folder",
    "Kolektor",
    "MVTec",  # Include MVTec for backward compatibility
    "MVTecAD",
    "VAD",
    "Visa",
    "Avenue",
    "ShanghaiTech",
    "UCSDped",
]
