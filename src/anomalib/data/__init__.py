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

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
from .datamodules.depth import DepthDataFormat, Folder3D, MVTec3D
from .datamodules.image import (
    VAD,
    BTech,
    Datumaro,
    Folder,
    ImageDataFormat,
    Kolektor,
    MVTec,
    MVTecAD,
    MVTecAD2,
    MVTecLOCO,
    RealIAD,
    Visa,
)
from .datamodules.video import Avenue, ShanghaiTech, UCSDped, VideoDataFormat

# Datasets
from .datasets import AnomalibDataset
from .datasets.depth import Folder3DDataset, MVTec3DDataset
from .datasets.image import (
    BTechDataset,
    DatumaroDataset,
    FolderDataset,
    KolektorDataset,
    MVTecADDataset,
    MVTecLOCODataset,
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
    logger.info("Loading the datamodule")

    if isinstance(config, dict):
        config = DictConfig(config)

    try:
        _config = config.data if "data" in config else config
        if len(_config.class_path.split(".")) > 1:
            module = importlib.import_module(".".join(_config.class_path.split(".")[:-1]))
        else:
            module = importlib.import_module("anomalib.data")
    except ModuleNotFoundError as exception:
        logger.exception(f"ModuleNotFoundError: {_config.class_path}")
        raise UnknownDatamoduleError from exception
    dataclass = getattr(module, _config.class_path.split(".")[-1])
    init_args = {**_config.get("init_args", {})}  # get dict
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
    # Image Data Modules
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
    "KolektorDataset",
    "MVTecADDataset",
    "MVTecLOCODataset",
    "VADDataset",
    "VisaDataset",
    "AvenueDataset",
    "ShanghaiTechDataset",
    "UCSDpedDataset",
    "PredictDataset",
    # Functions
    "get_datamodule",
    # Exceptions
    "UnknownDatamoduleError",
]
