"""Anomalib Datasets."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
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

    if config.dataset.format.lower() == "mvtec":
        datamodule = MVTec(
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
        datamodule = Folder(
            root=config.dataset.path,
            normal_dir=config.dataset.normal_dir,
            abnormal_dir=config.dataset.abnormal_dir,
            task=config.dataset.task,
            normal_test_dir=config.dataset.normal_test_dir,
            mask_dir=config.dataset.mask,
            extensions=config.dataset.extensions,
            split_ratio=config.dataset.split_ratio,
            seed=config.project.seed,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            train_batch_size=config.dataset.train_batch_size,
            test_batch_size=config.dataset.test_batch_size,
            num_workers=config.dataset.num_workers,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_val=config.dataset.transform_config.val,
            create_validation_set=config.dataset.create_validation_set,
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
