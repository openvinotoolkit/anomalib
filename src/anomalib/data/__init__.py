"""Anomalib Datasets."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import importlib
import logging
from enum import Enum

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


class DataFormat(str, Enum):
    """Supported Dataset Types"""

    MVTEC = "mvtec"
    MVTEC_3D = "mvtec_3d"
    BTECH = "btech"
    FOLDER = "folder"
    FOLDER_3D = "folder_3d"
    UCSDPED = "ucsdped"
    AVENUE = "avenue"
    VISA = "visa"
    SHANGHAITECH = "shanghaitech"


def get_datamodule(config: DictConfig | ListConfig) -> AnomalibDataModule:
    """Get Anomaly Datamodule.

    Args:
        config (DictConfig | ListConfig): Configuration of the anomaly model.

    Returns:
        PyTorch Lightning DataModule
    """
    logger.info("Loading the datamodule")

    datamodule: AnomalibDataModule

    module = importlib.import_module(".".join(config.data.class_path.split(".")[:-1]))
    dataclass = getattr(module, config.data.class_path.split(".")[-1])
    init_args = {**config.data.get("init_args", {})}  # get dict
    if "image_size" in init_args:
        init_args["image_size"] = to_tuple(init_args["image_size"])

    datamodule = dataclass(**init_args)

    return datamodule


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
