# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Anomalib Data Modules."""

from .depth import Folder3D, MVTec3D
from .image import (
    BMAD,
    MPDD,
    VAD,
    AutoVI,
    BTech,
    Datumaro,
    Folder,
    Kolektor,
    MVTecAD,
    Tabular,
    Visa,
)
from .video import Avenue, ShanghaiTech, UCSDped

__all__ = [
    "Folder3D",
    "MVTec3D",
    "BTech",
    "Datumaro",
    "Folder",
    "Kolektor",
    "MPDD",
    "MVTecAD",
    "Tabular",
    "VAD",
    "Visa",
    "Avenue",
    "ShanghaiTech",
    "UCSDped",
    "BMAD",
    "AutoVI",
]
