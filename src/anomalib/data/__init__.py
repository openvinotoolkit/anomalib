"""Anomalib Datasets."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import importlib
import logging
from enum import Enum
from itertools import chain

from omegaconf import DictConfig, ListConfig

from anomalib.utils.config import to_tuple

from .base import AnomalibDataModule, AnomalibDataset
from .depth import DepthDataFormat, Folder3D, MVTec3D
from .image import BTech, Folder, ImageDataFormat, Kolektor, MVTec, MVTecLoco, Visa
from .predict import PredictDataset
from .video import Avenue, ShanghaiTech, UCSDped, VideoDataFormat

logger = logging.getLogger(__name__)


DataFormat = Enum(  # type: ignore[misc]
    "DataFormat",
    {i.name: i.value for i in chain(DepthDataFormat, ImageDataFormat, VideoDataFormat)},
)


def get_datamodule(config: DictConfig | ListConfig) -> AnomalibDataModule:
    """Get Anomaly Datamodule.

    Args:
        config (DictConfig | ListConfig): Configuration of the anomaly model.

    Returns:
        PyTorch Lightning DataModule
    """
    logger.info("Loading the datamodule")

    module = importlib.import_module(".".join(config.data.class_path.split(".")[:-1]))
    dataclass = getattr(module, config.data.class_path.split(".")[-1])
    init_args = {**config.data.get("init_args", {})}  # get dict
    if "image_size" in init_args:
        init_args["image_size"] = to_tuple(init_args["image_size"])

    return dataclass(**init_args)


__all__ = [
    "AnomalibDataset",
    "AnomalibDataModule",
    "DepthDataFormat",
    "ImageDataFormat",
    "VideoDataFormat",
    "get_datamodule",
    "BTech",
    "Folder",
    "Folder3D",
    "PredictDataset",
    "Kolektor",
    "MVTec",
    "MVTec3D",
    "MVTecLoco",
    "Avenue",
    "UCSDped",
    "ShanghaiTech",
    "Visa",
]
