"""Anomalib dataset and datamodule base classes."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import Optional

from pandas import DataFrame
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset


class AnomalibDataset(Dataset, ABC):
    """Base Anomalib dataset."""

    def __init__(self, samples: DataFrame):
        super().__init__()
        self.samples = samples

    def contains_anomalous_images(self):
        """Check if the dataset contains any anomalous images."""
        return "anomalous" in list(self.samples.label)


class AnomalibDataModule(LightningDataModule):
    """Base Anomalib data module."""

    def __init__(self):
        super().__init__()
        self.train_data: Optional[AnomalibDataset] = None
        self.val_data: Optional[AnomalibDataset] = None
        self.test_data: Optional[AnomalibDataset] = None
