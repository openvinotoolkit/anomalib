"""Anomalib Datasets."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import importlib
import logging
from enum import Enum
from itertools import chain

from omegaconf import DictConfig, ListConfig

from anomalib.config.utils import to_tuple

from .avenue import Avenue
from .base import AnomalibDataModule, AnomalibDataset
from .btech import BTech
from .folder import Folder
from .folder_3d import Folder3D
from .inference import InferenceDataset
from .mvtec import MVTec
from .mvtec_3d import MVTec3D
from .shanghaitech import ShanghaiTech
from .task_type import TaskType
from .ucsd_ped import UCSDped
from .visa import Visa

logger = logging.getLogger(__name__)


class ImageDataFormat(str, Enum):
    """Supported Image Dataset Types."""

    MVTEC = "mvtec"
    MVTEC_3D = "mvtec_3d"
    BTECH = "btech"
    KOLEKTOR = "kolektor"
    FOLDER = "folder"
    FOLDER_3D = "folder_3d"
    VISA = "visa"


class VideoDataFormat(str, Enum):
    """Supported Video Dataset Types."""

    UCSDPED = "ucsdped"
    AVENUE = "avenue"
    SHANGHAITECH = "shanghaitech"


DataFormat = Enum(  # type: ignore[misc]
    "DataFormat",
    {i.name: i.value for i in chain(ImageDataFormat, VideoDataFormat)},
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
    "get_datamodule",
    "BTech",
    "Folder",
    "Folder3D",
    "InferenceDataset",
    "MVTec",
    "MVTec3D",
    "Avenue",
    "UCSDped",
    "TaskType",
    "ShanghaiTech",
    "Visa",
]
