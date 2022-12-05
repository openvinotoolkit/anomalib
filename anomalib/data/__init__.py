"""Anomalib Datasets."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Union

from omegaconf import DictConfig, ListConfig

from .avenue import Avenue
from .base import AnomalibDataModule, AnomalibDataset
from .btech import BTech
from .folder import Folder
from .inference import InferenceDataset
from .mvtec import MVTec
from .ucsd_ped import UCSDped

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
        datamodule = MVTec(
            root=config.dataset.path,
            category=config.dataset.category,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            task=config.dataset.task,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            val_split_mode=config.dataset.val_split_mode,
            val_split_ratio=config.dataset.val_split_ratio,
        )
    elif config.dataset.format.lower() == "btech":
        datamodule = BTech(
            root=config.dataset.path,
            category=config.dataset.category,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            task=config.dataset.task,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            val_split_mode=config.dataset.val_split_mode,
            val_split_ratio=config.dataset.val_split_ratio,
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
            normal_split_ratio=config.dataset.normal_split_ratio,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            val_split_mode=config.dataset.val_split_mode,
            val_split_ratio=config.dataset.val_split_ratio,
        )
    elif config.dataset.format.lower() == "ucsdped":
        datamodule = UCSDped(
            root=config.dataset.path,
            category=config.dataset.category,
            task=config.dataset.task,
            clip_length_in_frames=config.dataset.clip_length_in_frames,
            frames_between_clips=config.dataset.frames_between_clips,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            val_split_mode=config.dataset.val_split_mode,
            val_split_ratio=config.dataset.val_split_ratio,
        )
    elif config.dataset.format.lower() == "avenue":
        datamodule = Avenue(
            root=config.dataset.path,
            gt_dir=config.dataset.gt_dir,
            task=config.dataset.task,
            clip_length_in_frames=config.dataset.clip_length_in_frames,
            frames_between_clips=config.dataset.frames_between_clips,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            val_split_mode=config.dataset.val_split_mode,
            val_split_ratio=config.dataset.val_split_ratio,
        )
    else:
        raise ValueError(
            "Unknown dataset! \n"
            "If you use a custom dataset make sure you initialize it in"
            "`get_datamodule` in `anomalib.data.__init__.py"
        )

    return datamodule


__all__ = [
    "AnomalibDataset",
    "AnomalibDataModule",
    "get_datamodule",
    "BTech",
    "Folder",
    "InferenceDataset",
    "MVTec",
    "Avenue",
    "UCSDped",
]
