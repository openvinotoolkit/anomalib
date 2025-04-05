"""Anomalib Image Data Modules.

This module contains data modules for loading and processing image datasets for
anomaly detection. The following data modules are available:

- ``BTech``: BTech Surface Defect Dataset
- ``Datumaro``: Dataset in Datumaro format (Intel Geti™ export)
- ``Folder``: Custom folder structure with normal/abnormal images
- ``Kolektor``: Kolektor Surface-Defect Dataset
- ``MVTecAD``: MVTec Anomaly Detection Dataset
- ``MVTecAD2``: MVTec Anomaly Detection Dataset 2
- ``MVTecLOCO``: MVTec LOCO Dataset with logical and structural anomalies
- ``VAD``: Valeo Anomaly Detection Dataset
- ``Visa``: Visual Anomaly Dataset

Example:
    Load the MVTec AD dataset::

        >>> from anomalib.data import MVTecAD
        >>> datamodule = MVTecAD(
        ...     root="./datasets/MVTecAD",
        ...     category="bottle"
        ... )
"""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

from .btech import BTech
from .datumaro import Datumaro
from .folder import Folder
from .kolektor import Kolektor
from .mvtec_loco import MVTecLOCO
from .mvtecad import MVTec, MVTecAD
from .mvtecad2 import MVTecAD2
from .realiad import RealIAD
from .vad import VAD
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
    - ``MVTEC_AD_2``: MVTec AD 2 Dataset
    - ``MVTEC_3D``: MVTec 3D AD Dataset
    - ``MVTEC_LOCO``: MVTec LOCO Dataset
    - ``REALIAD``: Real-IAD Dataset
    - ``VAD``: Valeo Anomaly Detection Dataset
    - ``VISA``: Visual Anomaly Dataset
    """

    BTECH = "btech"
    DATUMARO = "datumaro"
    FOLDER = "folder"
    FOLDER_3D = "folder_3d"
    KOLEKTOR = "kolektor"
    MVTEC_AD = "mvtecad"
    MVTEC_AD_2 = "mvtecad2"
    MVTEC_3D = "mvtec_3d"
    MVTEC_LOCO = "mvtec_loco"
    REAL_IAD = "realiad"
    VAD = "vad"
    VISA = "visa"


__all__ = [
    "BTech",
    "Datumaro",
    "Folder",
    "Kolektor",
    "MVTec",  # Include MVTec for backward compatibility
    "MVTecAD",
    "MVTecAD2",
    "MVTecLOCO",
    "RealIAD",
    "VAD",
    "Visa",
]
