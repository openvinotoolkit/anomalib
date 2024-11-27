"""Anomalib Image Data Modules."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from .btech import BTech
from .csv import CSV
from .datumaro import Datumaro
from .folder import Folder
from .kolektor import Kolektor
from .mvtec import MVTec
from .visa import Visa


class ImageDataFormat(str, Enum):
    """Supported Image Dataset Types."""

    BTECH = "btech"
    CSV = "csv"
    DATUMARO = "datumaro"
    FOLDER = "folder"
    FOLDER_3D = "folder_3d"
    KOLEKTOR = "kolektor"
    MVTEC = "mvtec"
    MVTEC_3D = "mvtec_3d"
    VISA = "visa"


__all__ = ["BTech", "CSV", "Datumaro", "Folder", "Kolektor", "MVTec", "Visa"]
