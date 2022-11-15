"""Anomalib Datasets."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from importlib import import_module
from typing import Union

from omegaconf import DictConfig, ListConfig
from pytorch_lightning import LightningDataModule

from .btech import BTech
from .folder import Folder
from .inference import InferenceDataset
from .mvtec import MVTec

logger = logging.getLogger(__name__)


def get_datamodule(config: Union[DictConfig, ListConfig]) -> LightningDataModule:
    """Get Anomaly Datamodule.

    Args:
        config (Union[DictConfig, ListConfig]): Configuration of the anomaly model.

    Returns:
        PyTorch Lightning DataModule
    """
    logger.info("Loading the datamodule")

    datamodule: LightningDataModule

    # Since in the new config based on LightningCLI, seed_everything is a separate parameter, we set the one in dataset
    # to point to it.
    if config.data.init_args.seed == 0:
        config.data.init_args.seed = config.seed_everything

    if isinstance(config, (ListConfig, DictConfig)):
        # Need to remove image_size from config as transforms checks image_size against int or tuple whereas image_size
        # is ListConfig.
        image_size = tuple(config.data.init_args.image_size)
        config.data.init_args.pop("image_size")

    try:
        module = import_module(".".join(config.data.class_path.split(".")[:-1]))
        datamodule = getattr(module, config.data.class_path.split(".")[-1])(
            image_size=image_size, **config.data.init_args
        )
    except ModuleNotFoundError as exception:
        logger.error("Could not find the datamodule class: %s", config.data.class_path)
        raise exception

    return datamodule


__all__ = [
    "get_datamodule",
    "BTech",
    "Folder",
    "InferenceDataset",
    "MVTec",
]
