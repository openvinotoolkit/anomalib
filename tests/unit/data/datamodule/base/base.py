"""Unit Tests - Base Image Datamodules."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from torch.utils.data import DataLoader

from anomalib.data import AnomalibDataModule


class _TestAnomalibDataModule:
    """Base test class for ``AnomalibDataModule``.

    This is a base class for testing the AnomalibDataModule. Since
    ``AnomalibDataModule`` has methods that are yet to be implemented, this base
    test class is not meant to be used directly.
    """

    @staticmethod
    @pytest.mark.parametrize("subset", ["train", "val", "test"])
    def test_datamodule_has_dataloader_attributes(datamodule: AnomalibDataModule, subset: str) -> None:
        """Test that the datamodule has the correct dataloader attributes."""
        dataloader = f"{subset}_dataloader"
        assert hasattr(datamodule, dataloader)
        assert isinstance(getattr(datamodule, dataloader)(), DataLoader)

    @staticmethod
    def test_datamodule_from_config(fxt_data_config_path: str) -> None:
        # 1. Wrong file path:
        with pytest.raises(FileNotFoundError):
            AnomalibDataModule.from_config(config_path="wrong_configs.yaml")

        # 2. Correct file path:
        datamodule = AnomalibDataModule.from_config(config_path=fxt_data_config_path)
        assert datamodule is not None
        assert isinstance(datamodule, AnomalibDataModule)

        # 3. Override batch_size & num_workers
        override_kwargs = {"data.train_batch_size": 1, "data.num_workers": 1}
        datamodule = AnomalibDataModule.from_config(config_path=fxt_data_config_path, **override_kwargs)
        assert datamodule.train_batch_size == 1
        assert datamodule.num_workers == 1
