"""Tests for the Anomalib Engine."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Resize, Transform

from anomalib import LearningType, TaskType
from anomalib.data import AnomalibDataModule, AnomalibDataset
from anomalib.engine import Engine
from anomalib.models import AnomalyModule


class DummyDataset(AnomalibDataset):
    """Dummy dataset for testing the setup_transform method."""

    def __init__(self, transform: Transform = None) -> None:
        super().__init__(TaskType.CLASSIFICATION, transform=transform)
        self.image = torch.rand(3, 10, 10)
        self._samples = None

    def _setup(self, _stage: str | None = None) -> None:
        self._samples = None

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return 1


class DummyModel(AnomalyModule):
    """Dummy model for testing the setup_transform method."""

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Linear(10, 10)

    @staticmethod
    def configure_transforms(image_size: tuple[int, int] | None = None) -> Transform:
        """Return a Resize transform."""
        if image_size is None:
            image_size = (256, 256)
        return Resize(image_size)

    @staticmethod
    def trainer_arguments() -> dict:
        """Return an empty dictionary."""
        return {}

    @staticmethod
    def learning_type() -> LearningType:
        """Return the learning type."""
        return LearningType.ZERO_SHOT


class DummyDataModule(AnomalibDataModule):
    """Dummy datamodule for testing the setup_transform method."""

    def __init__(
        self,
        transform: Transform | None = None,
        train_transform: Transform | None = None,
        eval_transform: Transform | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=1,
            eval_batch_size=1,
            num_workers=0,
            val_split_mode="from_test",
            val_split_ratio=0.5,
            image_size=image_size,
            transform=transform,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

    def _create_val_split(self) -> None:
        pass

    def _create_test_split(self) -> None:
        pass

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = DummyDataset(transform=self.train_transform)
        self.val_data = DummyDataset(transform=self.eval_transform)
        self.test_data = DummyDataset(transform=self.eval_transform)


@pytest.fixture()
def checkpoint_path() -> Generator:
    """Fixture to create a temporary checkpoint file that stores a Resize transform."""
    # Create a temporary file
    transform = Resize((50, 50))
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "model.ckpt"
        checkpoint = {"transform": transform}
        torch.save(checkpoint, file_path)

        yield file_path


class TestSetupTransform:
    """Tests for the `_setup_transform` method of the Anomalib Engine."""

    # test update single dataloader
    @staticmethod
    def test_single_dataloader_default_transform() -> None:
        """Tests if the default model transform is used when no transform is passed to the dataloader."""
        dataset = DummyDataset()
        dataloader = DataLoader(dataset, batch_size=1)
        model = DummyModel()
        # before the setup_transform is called, the dataset should not have a transform
        assert dataset.transform is None
        Engine._setup_transform(model, dataloaders=dataloader)  # noqa: SLF001
        # after the setup_transform is called, the dataset should have the default transform from the model
        assert dataset.transform is not None

    # test update multiple dataloaders
    @staticmethod
    def test_multiple_dataloaders_default_transform() -> None:
        """Tests if the default model transform is used when no transform is passed to the dataloader."""
        dataset = DummyDataset()
        dataloader = DataLoader(dataset, batch_size=1)
        model = DummyModel()
        # before the setup_transform is called, the dataset should not have a transform
        assert dataset.transform is None
        Engine._setup_transform(model, dataloaders=[dataloader, dataloader])  # noqa: SLF001
        # after the setup_transform is called, the dataset should have the default transform from the model
        assert dataset.transform is not None

    @staticmethod
    def test_single_dataloader_custom_transform() -> None:
        """Tests if the user-specified transform is used when passed to the dataloader."""
        transform = Transform()
        dataset = DummyDataset(transform=transform)
        dataloader = DataLoader(dataset, batch_size=1)
        model = DummyModel()
        # before the setup_transform is called, the dataset should have the custom transform
        assert dataset.transform == transform
        Engine._setup_transform(model, dataloaders=dataloader)  # noqa: SLF001
        # after the setup_transform is called, the model should have the custom transform
        assert model.transform == transform

    # test if the user-specified transform is used when passed to the datamodule
    @staticmethod
    def test_custom_transform() -> None:
        """Tests if the user-specified transform is used when passed to the datamodule."""
        transform = Transform()
        datamodule = DummyDataModule(transform=transform)
        model = DummyModel()
        # assert that the datamodule uses the custom transform before and after setup_transform is called
        assert datamodule.train_transform == transform
        assert datamodule.eval_transform == transform
        Engine._setup_transform(model, datamodule=datamodule)  # noqa: SLF001
        assert datamodule.train_transform == transform
        assert datamodule.eval_transform == transform
        assert model.transform == transform

    # test if the user-specified transform is used when passed to the datamodule
    @staticmethod
    def test_custom_train_transform() -> None:
        """Tests if the user-specified transform is used when passed to the datamodule as train_transform."""
        model = DummyModel()
        transform = Transform()
        datamodule = DummyDataModule(train_transform=transform)
        # before calling setup, train_transform should be the custom transform and eval_transform should be None
        assert datamodule.train_transform == transform
        assert datamodule.eval_transform is None
        Engine._setup_transform(model, datamodule=datamodule)  # noqa: SLF001
        # after calling setup, train_transform should be the custom transform and eval_transform should be the default
        assert datamodule.train_transform == transform
        assert datamodule.eval_transform is None
        assert model.transform != transform
        assert model.transform is not None

    # test if the user-specified transform is used when passed to the datamodule
    @staticmethod
    def test_custom_eval_transform() -> None:
        """Tests if the user-specified transform is used when passed to the datamodule as eval_transform."""
        model = DummyModel()
        transform = Transform()
        datamodule = DummyDataModule(eval_transform=transform)
        # before calling setup, train_transform should be the custom transform and eval_transform should be None
        assert datamodule.train_transform is None
        assert datamodule.eval_transform == transform
        Engine._setup_transform(model, datamodule=datamodule)  # noqa: SLF001
        # after calling setup, train_transform should be the custom transform and eval_transform should be the default
        assert datamodule.train_transform is None
        assert datamodule.eval_transform == transform
        assert model.transform == transform

    # test update datamodule
    @staticmethod
    def test_datamodule_default_transform() -> None:
        """Tests if the default model transform is used when no transform is passed to the datamodule."""
        datamodule = DummyDataModule()
        model = DummyModel()
        # assert that the datamodule has a transform after the setup_transform is called
        Engine._setup_transform(model, datamodule=datamodule)  # noqa: SLF001
        assert isinstance(model.transform, Transform)

    # test if image size is taken from datamodule
    @staticmethod
    def test_datamodule_image_size() -> None:
        """Tests if the image size that is passed to the datamodule overwrites the default size from the model."""
        datamodule = DummyDataModule(image_size=(100, 100))
        model = DummyModel()
        # assert that the datamodule has a transform after the setup_transform is called
        Engine._setup_transform(model, datamodule=datamodule)  # noqa: SLF001
        assert isinstance(model.transform, Resize)
        assert model.transform.size == [100, 100]

    @staticmethod
    def test_transform_from_checkpoint(checkpoint_path: Path) -> None:
        """Tests if the transform from the checkpoint is used."""
        model = DummyModel()
        Engine._setup_transform(model, ckpt_path=checkpoint_path)  # noqa: SLF001
        assert isinstance(model.transform, Resize)
        assert model.transform.size == [50, 50]

    @staticmethod
    def test_precendence_datamodule(checkpoint_path: Path) -> None:
        """Tests if transform from the datamodule goes first if both checkpoint and datamodule are provided."""
        transform = Transform()
        datamodule = DummyDataModule(transform=transform)
        model = DummyModel()
        Engine._setup_transform(model, ckpt_path=checkpoint_path, datamodule=datamodule)  # noqa: SLF001
        assert model.transform == transform

    @staticmethod
    def test_transform_already_assigned() -> None:
        """Tests if the transform from the model is used when the model already has a transform assigned."""
        transform = Transform()
        model = DummyModel()
        model.set_transform(transform)
        datamodule = DummyDataModule()
        Engine._setup_transform(model, datamodule=datamodule)  # noqa: SLF001
        assert model.transform == transform
