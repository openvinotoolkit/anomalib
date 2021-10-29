"""
Anomaly Dataset Utils
"""

# Copyright (C) 2021 Intel Corporation
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


from typing import Dict, List, Optional, Union

from omegaconf import DictConfig, ListConfig
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.subset import Subset
from pytorch_lightning.core.datamodule import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from anomalib.datasets.transforms import PreProcessor


class OTEAnomalyDataset(Dataset):
    """
    Anomaly Dataset Adaptor
    This class converts OTE Dataset into Anomalib dataset that
    is a sub-class of Vision Dataset.

    Args:
        config (Union[DictConfig, ListConfig]): Anomalib config
        dataset (DatasetEntity): [description]: OTE SDK Dataset
        is_train (bool, optional): Train or Test mode. Defaults to True.

    Example:
        >>> from apis.tests.helpers.dataset import OTEAnomalyDatasetGenerator
        >>> from apis.ote.utils.data import AnomalyDataset

        >>> dataset_generator = OTEAnomalyDatasetGenerator()
        >>> dataset = dataset_generator.generate()
        >>> anomaly_dataset = AnomalyDataset(config=config, dataset=dataset)
        >>> anomaly_dataset[0]["image].shape
        torch.Size([3, 256, 256])
    """

    def __init__(self, config: Union[DictConfig, ListConfig], dataset: DatasetEntity):
        self.config = config
        self.dataset = dataset
        self.pre_processor = PreProcessor(config=self.config.transform)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Union[int, Tensor]]:
        item = self.dataset[index]
        image = self.pre_processor(image=item.numpy)["image"]
        try:
            label = 0 if item.get_shapes_labels()[0].name == "normal" else 1
        except IndexError:
            return {"index": index, "image": image}
        return {"index": index, "image": image, "label": label}


class OTEAnomalyDataModule(LightningDataModule):
    """
    Anomaly DataModule
    This class converts OTE Dataset into Anomalib dataset and stores
    train/val/test dataloaders.

    Args:
        config (Union[DictConfig, ListConfig]): Anomalib config
        dataset (DatasetEntity): [description]: OTE SDK Dataset
        is_train (bool, optional): Train or Test mode. Defaults to True.

    Example:
        >>> from apis.tests.helpers.dataset import OTEAnomalyDatasetGenerator
        >>> from apis.ote.utils.data import AnomalyDataModule

        >>> dataset_generator = OTEAnomalyDatasetGenerator()
        >>> dataset = dataset_generator.generate()
        >>> data_module = AnomalyDataModule(config=config, dataset=dataset)
        >>> i, data = next(enumerate(data_module.train_dataloader()))
        >>> data["image"].shape
        torch.Size([32, 3, 256, 256])
    """

    def __init__(self, config: Union[DictConfig, ListConfig], dataset: DatasetEntity) -> None:
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.pre_processor = PreProcessor(config=self.config.transform)

        self.train_ote_dataset: DatasetEntity
        self.val_ote_dataset: DatasetEntity
        self.test_ote_dataset: DatasetEntity
        self.predict_ote_dataset: DatasetEntity

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup Anomaly Data Module

        Args:
            stage (Optional[str], optional): train/val/test stages.
                Defaults to None.
        """
        if stage == "fit" or stage is None:
            self.train_ote_dataset = self.dataset.get_subset(Subset.TRAINING)
            self.val_ote_dataset = self.dataset.get_subset(Subset.VALIDATION)

        if stage == "validate":
            self.val_ote_dataset = self.dataset.get_subset(Subset.VALIDATION)

        if stage == "test" or stage is None:
            self.test_ote_dataset = self.dataset.get_subset(Subset.TESTING)

        if stage == "predict":
            self.predict_ote_dataset = self.dataset

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Train Dataloader
        """

        dataset = OTEAnomalyDataset(self.config, self.train_ote_dataset)
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.config.dataset.train_batch_size,
            num_workers=self.config.dataset.num_workers,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """
        Validation Dataloader
        """

        dataset = OTEAnomalyDataset(self.config, self.val_ote_dataset)
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.config.dataset.test_batch_size,
            num_workers=self.config.dataset.num_workers,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """
        Test Dataloader
        """
        dataset = OTEAnomalyDataset(self.config, self.test_ote_dataset)
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.config.dataset.test_batch_size,
            num_workers=self.config.dataset.num_workers,
        )

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        """
        Predict Dataloader
        """
        dataset = OTEAnomalyDataset(self.config, self.predict_ote_dataset)
        return DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.config.dataset.test_batch_size,
            num_workers=self.config.dataset.num_workers,
        )
