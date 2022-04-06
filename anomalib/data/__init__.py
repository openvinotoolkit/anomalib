"""Anomalib Datasets."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Union

from omegaconf import DictConfig, ListConfig
from pytorch_lightning import LightningDataModule

from .btech import BTechDataModule
from .folder import FolderDataModule
from .inference import InferenceDataset
from .mvtec import MVTecDataModule


def get_datamodule(config: Union[DictConfig, ListConfig]) -> LightningDataModule:
    """Get Anomaly Datamodule.

    Args:
        config (Union[DictConfig, ListConfig]): Configuration of the anomaly model.

    Returns:
        PyTorch Lightning DataModule
    """
    datamodule: LightningDataModule

    if config.dataset.format.lower() == "mvtec":
        datamodule = MVTecDataModule(
            # TODO: Remove config values. IAAALD-211
            root=config.dataset.path,
            category=config.dataset.category,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            train_batch_size=config.dataset.train_batch_size,
            test_batch_size=config.dataset.test_batch_size,
            num_workers=config.dataset.num_workers,
            seed=config.project.seed,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_val=config.dataset.transform_config.val,
            create_validation_set=config.dataset.create_validation_set,
        )
    elif config.dataset.format.lower() == "btech":
        datamodule = BTechDataModule(
            # TODO: Remove config values. IAAALD-211
            root=config.dataset.path,
            category=config.dataset.category,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
            train_batch_size=config.dataset.train_batch_size,
            test_batch_size=config.dataset.test_batch_size,
            num_workers=config.dataset.num_workers,
            seed=config.project.seed,
            transform_config_train=config.dataset.transform_config.train,
            transform_config_val=config.dataset.transform_config.val,
            create_validation_set=config.dataset.create_validation_set,
        )
    elif config.dataset.format.lower() == "folder":
        datamodule = FolderDataModule(
            root=config.dataset.path,
            normal=config.dataset.normal,
            abnormal=config.dataset.abnormal,
            task=config.dataset.task,
            mask_dir=config.dataset.mask,
            extensions=config.dataset.extensions,
            split_ratio=config.dataset.split_ratio,
            seed=config.dataset.seed,
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
    "BTechDataModule",
    "FolderDataModule",
    "InferenceDataset",
    "MVTecDataModule",
]
