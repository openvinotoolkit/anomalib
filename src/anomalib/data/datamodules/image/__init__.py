"""Anomalib Image Data Modules.

This module contains data modules for loading and processing image datasets for
anomaly detection. The following data modules are available:

- ``BTech``: BTech Surface Defect Dataset
- ``Datumaro``: Dataset in Datumaro format (Intel Geti™ export)
- ``Folder``: Custom folder structure with normal/abnormal images
- ``Kolektor``: Kolektor Surface-Defect Dataset
- ``MVTecAD``: MVTec Anomaly Detection Dataset
- ``Visa``: Visual Inspection for Steel Anomaly Dataset

Example:
    Load the MVTec AD dataset::

        >>> from anomalib.data import MVTecAD
        >>> datamodule = MVTecAD(
        ...     root="./datasets/MVTecAD",
        ...     category="bottle"
        ... )
"""

# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from .btech import BTech
from .datumaro import Datumaro
from .folder import Folder
from .kolektor import Kolektor
from .mvtecad import MVTec, MVTecAD
from .visa import Visa


class ImageDataFormat(str, Enum):
    """Supported Image Dataset Types.

    The following dataset formats are supported:

    - ``BTECH``: BTech Surface Defect Dataset
    - ``DATUMARO``: Dataset in Datumaro format
    - ``FOLDER``: Custom folder structure
    - ``FOLDER_3D``: Custom folder structure for 3D images
    - ``KOLEKTOR``: Kolektor Surface-Defect Dataset
    - ``MVTEC_AD``: MVTec AD Dataset
    - ``MVTEC_3D``: MVTec 3D AD Dataset
    - ``VISA``: Visual Inspection for Steel Anomaly Dataset
    """

    BTECH = "btech"
    DATUMARO = "datumaro"
    FOLDER = "folder"
    FOLDER_3D = "folder_3d"
    KOLEKTOR = "kolektor"
    MVTEC_AD = "mvtecad"
    MVTEC_3D = "mvtec_3d"
    VISA = "visa"


__all__ = [
    "BTech",
    "Datumaro",
    "Folder",
    "Kolektor",
    "MVTecAD",
    "MVTec",  # Include both for backward compatibility
    "Visa",
]
