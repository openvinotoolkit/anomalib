"""Anomalib Image Datasets.

This module contains the supported image datasets for Anomalib.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from .btech import BTech
from .dataframe import Dataframe
from .datumaro import Datumaro
from .folder import Folder
from .kolektor import Kolektor
from .mvtec import MVTec
from .visa import Visa


class ImageDataFormat(str, Enum):
    """Supported Image Dataset Types."""

    BTECH = "btech"
    DATAFRAME = "dataframe"
    DATUMARO = "datumaro"
    FOLDER = "folder"
    FOLDER_3D = "folder_3d"
    KOLEKTOR = "kolektor"
    MVTEC = "mvtec"
    MVTEC_3D = "mvtec_3d"
    VISA = "visa"


__all__ = ["BTech", "Dataframe", "Datumaro", "Folder", "Kolektor", "MVTec", "Visa"]
