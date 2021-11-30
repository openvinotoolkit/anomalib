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

from .anomaly_dataset import AnomalyDataModule
from .mvtec import MVTecDataModule


def get_datamodule(config: Union[DictConfig, ListConfig]):
    """Get Anomaly Datamodule.

    Args:
        config: Configuration of the anomaly model
        config: Union[DictConfig, ListConfig]:

    Returns:
        PyTorch Lightning DataModule
    """
    datamodule: LightningDataModule

    if config.dataset.format.lower() == "mvtec":
        datamodule = MVTecDataModule(
            root=config.dataset.path,
            category=config.dataset.category,
            image_size=config.dataset.image_size,
            crop_size=config.dataset.image_size,
            train_batch_size=config.dataset.train_batch_size,
            test_batch_size=config.dataset.test_batch_size,
            num_workers=config.dataset.num_workers,
        )
    elif config.dataset.format.lower() == "anomaly_dataset":
        datamodule = AnomalyDataModule(
            root=config.dataset.path,
            url=config.dataset.url,
            category=config.dataset.category,
            task=config.dataset.task,
            label_format=config.dataset.label_format,
            train_batch_size=config.dataset.train_batch_size,
            test_batch_size=config.dataset.test_batch_size,
            num_workers=config.dataset.num_workers,
            image_size=(config.dataset.image_size[0], config.dataset.image_size[1]),
        )
    else:
        raise ValueError("Unknown dataset!")

    return datamodule
