"""Anomalib Datasets."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import importlib
import logging
from enum import Enum
from itertools import chain

from omegaconf import DictConfig, ListConfig

from anomalib.utils.config import to_tuple

from .base import AnomalibDataModule, AnomalibDataset
from .depth import DepthDataFormat, Folder3D, MVTec3D
from .image import BTech, Folder, ImageDataFormat, Kolektor, MVTec, Visa
from .predict import PredictDataset
from .utils import LabelName
from .video import Avenue, ShanghaiTech, UCSDped, VideoDataFormat

logger = logging.getLogger(__name__)


DataFormat = Enum(  # type: ignore[misc]
    "DataFormat",
    {i.name: i.value for i in chain(DepthDataFormat, ImageDataFormat, VideoDataFormat)},
)


class UnknownDatamoduleError(ModuleNotFoundError):
    ...


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
        if len(config.data.class_path.split(".")) > 1:
            module = importlib.import_module(".".join(config.data.class_path.split(".")[:-1]))
        else:
            module = importlib.import_module("anomalib.data")
    except ModuleNotFoundError as exception:
        logger.exception(f"ModuleNotFoundError: {config.data.class_path}")
        raise UnknownDatamoduleError from exception
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
    "Avenue",
    "UCSDped",
    "ShanghaiTech",
    "Visa",
    "LabelName",
]
