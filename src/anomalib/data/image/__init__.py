"""Anomalib Image Datasets.

This module contains the supported image datasets for Anomalib.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from .btech import BTech
from .csv import CSV
from .folder import Folder
from .kolektor import Kolektor
from .mvtec import MVTec
from .visa import Visa


class ImageDataFormat(str, Enum):
    """Supported Image Dataset Types."""

    BTECH = "btech"
    CSV = "csv"
    FOLDER = "folder"
    FOLDER_3D = "folder_3d"
    KOLEKTOR = "kolektor"
    MVTEC = "mvtec"
    MVTEC_3D = "mvtec_3d"
    VISA = "visa"


__all__ = ["BTech", "CSV", "Folder", "Kolektor", "MVTec", "Visa"]
