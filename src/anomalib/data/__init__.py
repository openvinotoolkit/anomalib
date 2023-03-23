"""Anomalib Datasets."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from omegaconf import DictConfig, ListConfig

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


def get_datamodule(config: DictConfig | ListConfig) -> AnomalibDataModule:
    """Get Anomaly Datamodule.

    Args:
        config (DictConfig | ListConfig): Configuration of the anomaly model.

    Returns:
        PyTorch Lightning DataModule
    """
    logger.info("Loading the datamodule")

    datamodule: AnomalibDataModule

    # convert center crop to tuple
    center_crop = config.dataset.get("center_crop")
    if center_crop is not None:
        center_crop = (center_crop[0], center_crop[1])

    if config.dataset.format.lower() == "mvtec":
        datamodule = MVTec(
            root=config.dataset.path,
            category=config.dataset.category,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            center_crop=center_crop,
            normalization=config.dataset.normalization,
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            task=config.dataset.task,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            test_split_mode=config.dataset.test_split_mode,
            test_split_ratio=config.dataset.test_split_ratio,
            val_split_mode=config.dataset.val_split_mode,
            val_split_ratio=config.dataset.val_split_ratio,
        )
    elif config.dataset.format.lower() == "mvtec_3d":
        datamodule = MVTec3D(
            root=config.dataset.path,
            category=config.dataset.category,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            center_crop=center_crop,
            normalization=config.dataset.normalization,
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            task=config.dataset.task,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            test_split_mode=config.dataset.test_split_mode,
            test_split_ratio=config.dataset.test_split_ratio,
            val_split_mode=config.dataset.val_split_mode,
            val_split_ratio=config.dataset.val_split_ratio,
        )
    elif config.dataset.format.lower() == "btech":
        datamodule = BTech(
            root=config.dataset.path,
            category=config.dataset.category,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            center_crop=center_crop,
            normalization=config.dataset.normalization,
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            task=config.dataset.task,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            test_split_mode=config.dataset.test_split_mode,
            test_split_ratio=config.dataset.test_split_ratio,
            val_split_mode=config.dataset.val_split_mode,
            val_split_ratio=config.dataset.val_split_ratio,
        )
    elif config.dataset.format.lower() == "folder":
        datamodule = Folder(
            root=config.dataset.root,
            normal_dir=config.dataset.normal_dir,
            abnormal_dir=config.dataset.abnormal_dir,
            task=config.dataset.task,
            normal_test_dir=config.dataset.normal_test_dir,
            mask_dir=config.dataset.mask_dir,
            extensions=config.dataset.extensions,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            center_crop=center_crop,
            normalization=config.dataset.normalization,
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            test_split_mode=config.dataset.test_split_mode,
            test_split_ratio=config.dataset.test_split_ratio,
            val_split_mode=config.dataset.val_split_mode,
            val_split_ratio=config.dataset.val_split_ratio,
        )
    elif config.dataset.format.lower() == "folder_3d":
        datamodule = Folder3D(
            root=config.dataset.root,
            normal_dir=config.dataset.normal_dir,
            normal_depth_dir=config.dataset.normal_depth_dir,
            abnormal_dir=config.dataset.abnormal_dir,
            abnormal_depth_dir=config.dataset.abnormal_depth_dir,
            task=config.dataset.task,
            normal_test_dir=config.dataset.normal_test_dir,
            normal_test_depth_dir=config.dataset.normal_test_depth_dir,
            mask_dir=config.dataset.mask_dir,
            extensions=config.dataset.extensions,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            center_crop=center_crop,
            normalization=config.dataset.normalization,
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            test_split_mode=config.dataset.test_split_mode,
            test_split_ratio=config.dataset.test_split_ratio,
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
            center_crop=center_crop,
            normalization=config.dataset.normalization,
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
            center_crop=center_crop,
            normalization=config.dataset.normalization,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            val_split_mode=config.dataset.val_split_mode,
            val_split_ratio=config.dataset.val_split_ratio,
        )
    elif config.dataset.format.lower() == "visa":
        datamodule = Visa(
            root=config.dataset.path,
            category=config.dataset.category,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            center_crop=center_crop,
            normalization=config.dataset.normalization,
            train_batch_size=config.dataset.train_batch_size,
            eval_batch_size=config.dataset.eval_batch_size,
            num_workers=config.dataset.num_workers,
            task=config.dataset.task,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_eval=config.dataset.transform_config.eval,
            test_split_mode=config.dataset.test_split_mode,
            test_split_ratio=config.dataset.test_split_ratio,
            val_split_mode=config.dataset.val_split_mode,
            val_split_ratio=config.dataset.val_split_ratio,
        )
    elif config.dataset.format.lower() == "shanghaitech":
        datamodule = ShanghaiTech(
            root=config.dataset.path,
            scene=config.dataset.scene,
            task=config.dataset.task,
            clip_length_in_frames=config.dataset.clip_length_in_frames,
            frames_between_clips=config.dataset.frames_between_clips,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            center_crop=center_crop,
            normalization=config.dataset.normalization,
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
    "Folder3D",
    "InferenceDataset",
    "MVTec",
    "MVTec3D",
    "Avenue",
    "UCSDped",
    "TaskType",
    "ShanghaiTech",
]
