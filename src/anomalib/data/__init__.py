# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Anomalib Datasets.

This module provides datasets and data modules for anomaly detection tasks.

The module contains:
    - Data classes for representing different types of data (images, videos, etc.)
    - Dataset classes for loading and processing data
    - Data modules for use with PyTorch Lightning
    - Helper functions for data loading and validation

Example:
    >>> from anomalib.data import MVTecAD
    >>> datamodule = MVTecAD(
    ...     root="./datasets/MVTecAD",
    ...     category="bottle",
    ...     image_size=(256, 256)
    ... )
"""

import importlib
import logging
from enum import Enum
from itertools import chain

from omegaconf import DictConfig, ListConfig

from anomalib.utils.config import to_tuple

# Dataclasses
from .dataclasses import (
    Batch,
    DatasetItem,
    DepthBatch,
    DepthItem,
    ImageBatch,
    ImageItem,
    InferenceBatch,
    NumpyImageBatch,
    NumpyImageItem,
    NumpyVideoBatch,
    NumpyVideoItem,
    VideoBatch,
    VideoItem,
)

# Datamodules
from .datamodules.base import AnomalibDataModule
from .datamodules.depth import ADAM3D, DepthDataFormat, Folder3D, MVTec3D
from .datamodules.image import (
    BMAD,
    MPDD,
    VAD,
    AutoVI,
    BTech,
    Datumaro,
    Folder,
    ImageDataFormat,
    Kaputt,
    Kolektor,
    MVTecAD,
    MVTecAD2,
    MVTecLOCO,
    RealIAD,
    Tabular,
    Visa,
)
from .datamodules.video import Avenue, ShanghaiTech, UCSDped, VideoDataFormat

# Datasets
from .datasets import AnomalibDataset
from .datasets.depth import ADAM3DDataset, Folder3DDataset, MVTec3DDataset
from .datasets.image import (
    AutoVIDataset,
    BMADDataset,
    BTechDataset,
    DatumaroDataset,
    FolderDataset,
    KaputtDataset,
    KolektorDataset,
    MPDDDataset,
    MVTecADDataset,
    MVTecLOCODataset,
    TabularDataset,
    VADDataset,
    VisaDataset,
)
from .datasets.video import AvenueDataset, ShanghaiTechDataset, UCSDpedDataset
from .predict import PredictDataset

logger = logging.getLogger(__name__)


DataFormat = Enum(  # type: ignore[misc]
    "DataFormat",
    {i.name: i.value for i in chain(DepthDataFormat, ImageDataFormat, VideoDataFormat)},
)


class UnknownDatamoduleError(ModuleNotFoundError):
    """Raised when a datamodule cannot be found."""


def get_datamodule(config: DictConfig | ListConfig | dict) -> AnomalibDataModule:
    """Get Anomaly Datamodule from config.

    Args:
        config: Configuration for the anomaly model. Can be either:
            - DictConfig from OmegaConf
            - ListConfig from OmegaConf
            - Python dictionary

    Returns:
        PyTorch Lightning DataModule configured according to the input.

    Raises:
        UnknownDatamoduleError: If the specified datamodule cannot be found.

    Example:
        >>> from omegaconf import DictConfig
        >>> config = DictConfig({
        ...     "data": {
        ...         "class_path": "MVTecAD",
        ...         "init_args": {"root": "./datasets/MVTec"}
        ...     }
        ... })
        >>> datamodule = get_datamodule(config)
    """
    logger.info("Loading the datamodule and dataset class from the config.")

    # Getting the datamodule class from the config.
    if isinstance(config, dict):
        config = DictConfig(config)
    config_ = config.data if "data" in config else config

    # All the sub data modules are imported to anomalib.data. So need to import the module dynamically using paths.
    module = importlib.import_module("anomalib.data")
    data_class_name = config_.class_path.split(".")[-1]
    # check if the data_class exists in the module
    if not hasattr(module, data_class_name):
        logger.error(
            f"Dataclass '{data_class_name}' not found in module '{module.__name__}'. "
            f"Available classes are {AnomalibDataModule.__subclasses__()}",
        )
        error_str = f"Dataclass '{data_class_name}' not found in module '{module.__name__}'."
        raise UnknownDatamoduleError(error_str)
    dataclass = getattr(module, data_class_name)

    init_args = {**config_.get("init_args", {})}  # get dict
    if "image_size" in init_args:
        init_args["image_size"] = to_tuple(init_args["image_size"])
    return dataclass(**init_args)


__all__ = [
    # Base Classes
    "AnomalibDataModule",
    "AnomalibDataset",
    # Data Classes
    "Batch",
    "DatasetItem",
    "DepthBatch",
    "DepthItem",
    "ImageBatch",
    "ImageItem",
    "InferenceBatch",
    "NumpyImageBatch",
    "NumpyImageItem",
    "NumpyVideoBatch",
    "NumpyVideoItem",
    "VideoBatch",
    "VideoItem",
    # Data Formats
    "DataFormat",
    "DepthDataFormat",
    "ImageDataFormat",
    "VideoDataFormat",
    # Depth Data Modules
    "Folder3D",
    "MVTec3D",
    "ADAM3D",
    # Image Data Modules
    "AutoVI",
    "BMAD",
    "BTech",
    "Datumaro",
    "Folder",
    "Kaputt",
    "Kolektor",
    "MPDD",
    "MVTecAD",
    "MVTecAD2",
    "MVTecLOCO",
    "RealIAD",
    "Tabular",
    "VAD",
    "Visa",
    # Video Data Modules
    "Avenue",
    "ShanghaiTech",
    "UCSDped",
    # Datasets
    "Folder3DDataset",
    "MVTec3DDataset",
    "BTechDataset",
    "DatumaroDataset",
    "FolderDataset",
    "KaputtDataset",
    "KolektorDataset",
    "MPDDDataset",
    "ADAM3DDataset",
    "MVTecADDataset",
    "MVTecLOCODataset",
    "TabularDataset",
    "VADDataset",
    "VisaDataset",
    "AvenueDataset",
    "ShanghaiTechDataset",
    "UCSDpedDataset",
    "PredictDataset",
    "BMADDataset",
    # Functions
    "get_datamodule",
    # Exceptions
    "UnknownDatamoduleError",
    "AutoVIDataset",
]
