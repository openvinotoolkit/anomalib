"""Anomalib Datasets."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Union

from omegaconf import DictConfig, ListConfig

from anomalib.data.base import AnomalibDataModule

from .btech import BTech
from .folder import FolderDataModule
from .inference import InferenceDataset
from .mvtec import MVTecDataModule

logger = logging.getLogger(__name__)


def get_datamodule(config: Union[DictConfig, ListConfig]) -> AnomalibDataModule:
    """Get Anomaly Datamodule.

    Args:
        config (Union[DictConfig, ListConfig]): Configuration of the anomaly model.

    Returns:
        PyTorch Lightning DataModule
    """
    logger.info("Loading the datamodule")

    datamodule: AnomalibDataModule

    if config.dataset.format.lower() == "mvtec":
        datamodule = MVTecDataModule(
            # TODO: Remove config values. IAAALD-211
            root=config.dataset.path,
            category=config.dataset.category,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            train_batch_size=config.dataset.train_batch_size,
            test_batch_size=config.dataset.test_batch_size,
            num_workers=config.dataset.num_workers,
            task=config.dataset.task,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_val=config.dataset.transform_config.val,
            val_split_mode=config.dataset.validation_split_mode,
        )
    elif config.dataset.format.lower() == "btech":
        datamodule = BTech(
            # TODO: Remove config values. IAAALD-211
            root=config.dataset.path,
            category=config.dataset.category,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            train_batch_size=config.dataset.train_batch_size,
            test_batch_size=config.dataset.test_batch_size,
            num_workers=config.dataset.num_workers,
            seed=config.project.seed,
            task=config.dataset.task,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_val=config.dataset.transform_config.val,
            create_validation_set=config.dataset.create_validation_set,
        )
    elif config.dataset.format.lower() == "folder":
        datamodule = FolderDataModule(
            root=config.dataset.path,
            normal_dir=config.dataset.normal_dir,
            abnormal_dir=config.dataset.abnormal_dir,
            task=config.dataset.task,
            normal_test_dir=config.dataset.normal_test_dir,
            mask_dir=config.dataset.mask,
            extensions=config.dataset.extensions,
            split_ratio=config.dataset.split_ratio,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            train_batch_size=config.dataset.train_batch_size,
            test_batch_size=config.dataset.test_batch_size,
            num_workers=config.dataset.num_workers,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_val=config.dataset.transform_config.val,
            val_split_mode=config.dataset.validation_split_mode,
        )
    else:
        raise ValueError(
            "Unknown dataset! \n"
            "If you use a custom dataset make sure you initialize it in"
            "`get_datamodule` in `anomalib.data.__init__.py"
        )

    return datamodule


__all__ = [
    "get_datamodule",
    "BTech",
    "Folder",
    "InferenceDataset",
    "MVTec",
]
