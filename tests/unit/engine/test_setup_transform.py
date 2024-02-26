"""Tests for the Anomalib Engine."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return the image tensor."""
        if self.transform:
            return self.transform(self.image)
        return self.image

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input tensor."""
        return self.model(x)

    def configure_transforms(self, image_size: tuple[int, int] | None = None) -> Transform:
        """Return a Resize transform."""
        if image_size is None:
            image_size = (256, 256)
        return Resize(image_size)

    def trainer_arguments(self) -> dict:
        """Return an empty dictionary."""
        return {}

    def learning_type(self) -> LearningType:
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


class TestSetupTransform:
    """Tests for the `_setup_transform` method of the Anomalib Engine."""

    # test update single dataloader
    def test_single_dataloader_default_transform(self) -> None:
        """Tests if the default model transform is used when no transform is passed to the dataloader."""
        engine = Engine()
        dataset = DummyDataset()
        dataloader = DataLoader(dataset, batch_size=1)
        model = DummyModel()
        # before the setup_transform is called, the dataset should not have a transform
        assert dataset.transform is None
        engine._setup_transform(model, dataloaders=dataloader)  # noqa: SLF001
        # after the setup_transform is called, the dataset should have the default transform from the model
        assert dataset.transform is not None

    # test update multiple dataloaders
    def test_multiple_dataloaders_default_transform(self) -> None:
        """Tests if the default model transform is used when no transform is passed to the dataloader."""
        engine = Engine()
        dataset = DummyDataset()
        dataloader = DataLoader(dataset, batch_size=1)
        model = DummyModel()
        # before the setup_transform is called, the dataset should not have a transform
        assert dataset.transform is None
        engine._setup_transform(model, dataloaders=[dataloader, dataloader])  # noqa: SLF001
        # after the setup_transform is called, the dataset should have the default transform from the model
        assert dataset.transform is not None

    # test if the user-specified transform is used when passed to the datamodule
    def test_user_specified_transform(self) -> None:
        """Tests if the user-specified transform is used when passed to the datamodule."""
        custom_transform = Resize((100, 100))
        engine = Engine()
        datamodule = DummyDataModule(transform=custom_transform)
        model = DummyModel()
        # assert that the datamodule uses the custom transform before and after setup_transform is called
        assert datamodule.train_transform == custom_transform
        assert datamodule.eval_transform == custom_transform
        engine._setup_transform(model, datamodule=datamodule)  # noqa: SLF001
        assert datamodule.train_transform == custom_transform
        assert datamodule.eval_transform == custom_transform

    # test if the user-specified transform is used when passed to the datamodule
    def test_user_specified_train_transform(self) -> None:
        """Tests if the user-specified transform is used when passed to the datamodule as train_transform."""
        model = DummyModel()
        default_size = model.configure_transforms().size
        custom_transform = Resize((100, 100))
        engine = Engine()
        datamodule = DummyDataModule(train_transform=custom_transform)
        # before calling setup, train_transform should be the custom transform and eval_transform should be None
        assert datamodule.train_transform == custom_transform
        assert datamodule.eval_transform is None
        engine._setup_transform(model, datamodule=datamodule)  # noqa: SLF001
        # after calling setup, train_transform should be the custom transform and eval_transform should be the default
        assert datamodule.train_transform == custom_transform
        assert isinstance(datamodule.eval_transform, Resize)
        assert datamodule.eval_transform.size == default_size

    # test update datamodule
    def test_datamodule_default_transform(self) -> None:
        """Tests if the default model transform is used when no transform is passed to the datamodule."""
        engine = Engine()
        datamodule = DummyDataModule()
        model = DummyModel()
        # assert that the datamodule has a transform after the setup_transform is called
        assert datamodule.train_transform is None
        assert datamodule.eval_transform is None
        engine._setup_transform(model, datamodule=datamodule)  # noqa: SLF001
        assert datamodule.train_transform is not None
        assert datamodule.eval_transform is not None

    # test if image size is taken from datamodule
    def test_datamodule_image_size(self) -> None:
        """Tests if the image size that is passed to the datamodule overwrites the default size from the model."""
        engine = Engine()
        datamodule = DummyDataModule(image_size=(100, 100))
        model = DummyModel()
        # assert that the datamodule has a transform after the setup_transform is called
        assert datamodule.train_transform is None
        assert datamodule.eval_transform is None
        engine._setup_transform(model, datamodule=datamodule)  # noqa: SLF001
        assert isinstance(datamodule.train_transform, Resize)
        assert isinstance(datamodule.eval_transform, Resize)
        assert datamodule.train_transform.size == [100, 100]
        assert datamodule.eval_transform.size == [100, 100]

    # test if the updated transform is used in the dataloader
    def test_datamodule_dataloader(self) -> None:
        """Tests if batch returned by the dataloader has the correct shape after the setup_transform is called."""
        engine = Engine()
        datamodule = DummyDataModule()
        model = DummyModel()
        engine._setup_transform(model, datamodule=datamodule)  # noqa: SLF001
        datamodule.setup()
        dataloader = datamodule.train_dataloader()
        for batch in dataloader:
            assert batch.shape == (1, 3, 256, 256)
