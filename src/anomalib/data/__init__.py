"""Anomalib Datasets."""

# Copyright (C) 2022-2024 Intel Corporation
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
from .datamodules.image import CSV, BTech, Datumaro, Folder, ImageDataFormat, Kolektor, MVTec, Visa
from .datamodules.video import Avenue, ShanghaiTech, UCSDped, VideoDataFormat

# Datasets
from .datasets import AnomalibDataset
from .datasets.depth import Folder3DDataset, MVTec3DDataset
from .datasets.image import BTechDataset, DatumaroDataset, FolderDataset, KolektorDataset, MVTecDataset, VisaDataset
from .datasets.video import AvenueDataset, ShanghaiTechDataset, UCSDpedDataset
from .predict import PredictDataset

logger = logging.getLogger(__name__)


DataFormat = Enum(  # type: ignore[misc]
    "DataFormat",
    {i.name: i.value for i in chain(DepthDataFormat, ImageDataFormat, VideoDataFormat)},
)


class UnknownDatamoduleError(ModuleNotFoundError): ...


def get_datamodule(config: DictConfig | ListConfig | dict) -> AnomalibDataModule:
    """Get Anomaly Datamodule.

    Args:
        config (DictConfig | ListConfig | dict): Configuration of the anomaly model.

    Returns:
        PyTorch Lightning DataModule
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
    # Anomalib dataclasses
    "DatasetItem",
    "Batch",
    "InferenceBatch",
    "ImageItem",
    "ImageBatch",
    "VideoItem",
    "VideoBatch",
    "DepthItem",
    "DepthBatch",
    "NumpyImageItem",
    "NumpyImageBatch",
    "NumpyVideoItem",
    "NumpyVideoBatch",
    # Anomalib datasets
    "AnomalibDataset",
    "Folder3DDataset",
    "MVTec3DDataset",
    "BTechDataset",
    "DatumaroDataset",
    "FolderDataset",
    "KolektorDataset",
    "MVTecDataset",
    "VisaDataset",
    "AvenueDataset",
    "ShanghaiTechDataset",
    "UCSDpedDataset",
    "PredictDataset",
    # Anomalib datamodules
    "AnomalibDataModule",
    "DepthDataFormat",
    "ImageDataFormat",
    "VideoDataFormat",
    "get_datamodule",
    "BTech",
    "CSV",
    "Datumaro",
    "Folder",
    "Folder3D",
    "Kolektor",
    "MVTec",
    "MVTec3D",
    "Avenue",
    "UCSDped",
    "ShanghaiTech",
    "Visa",
    "PredictDataset",
]
