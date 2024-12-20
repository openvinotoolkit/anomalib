"""Tests for the _update_subset_augmentations method in AnomalibDataModule."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
from torchvision.transforms.v2 import (
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
)

from anomalib.data import AnomalibDataModule, AnomalibDataset


class DummyDataset(AnomalibDataset):
    """Dummy dataset class for testing."""

    def __init__(self) -> None:
        pass


class TestUpdateAugmentations:
    """Tests for the _update_subset_augmentations method in AnomalibDataModule."""

    @staticmethod
    def test_conflicting_shape(caplog: pytest.LogCaptureFixture) -> None:
        """Test that a warning is logged if resize shapes mismatch."""
        dataset = DummyDataset()
        model_transform = Resize((224, 224))
        augmentations = Resize((256, 256))

        with caplog.at_level(logging.WARNING):
            AnomalibDataModule._update_subset_augmentations(dataset, augmentations, model_transform=model_transform)  # noqa: SLF001
        # check if a warning was logged
        assert any(record.levelname == "WARNING" for record in caplog.records)
        assert "Conflicting resize shape" in caplog.text
        # check if augmentations were overwritten by model transform
        assert dataset.augmentations == model_transform

    @staticmethod
    def test_conflicting_interpolation(caplog: pytest.LogCaptureFixture) -> None:
        """Test that a warning is logged if interpolation methods mismatch."""
        dataset = DummyDataset()
        model_transform = Resize((224, 224), interpolation=InterpolationMode.BILINEAR)
        augmentations = Resize((224, 224), interpolation=InterpolationMode.NEAREST)

        with caplog.at_level(logging.WARNING):
            AnomalibDataModule._update_subset_augmentations(dataset, augmentations, model_transform=model_transform)  # noqa: SLF001
        # check if a warning was logged
        assert any(record.levelname == "WARNING" for record in caplog.records)
        assert "Conflicting interpolation method" in caplog.text
        # check if augmentations were overwritten by model transform
        assert dataset.augmentations == model_transform

    @staticmethod
    def test_conflicting_antialias(caplog: pytest.LogCaptureFixture) -> None:
        """Test that a warning is logged if antialiasing setting mismatch."""
        dataset = DummyDataset()
        model_transform = Resize((224, 224), antialias=True)
        augmentations = Resize((224, 224), antialias=False)

        with caplog.at_level(logging.WARNING):
            AnomalibDataModule._update_subset_augmentations(dataset, augmentations, model_transform=model_transform)  # noqa: SLF001
        # check if a warning was logged
        assert any(record.levelname == "WARNING" for record in caplog.records)
        assert "Conflicting antialiasing setting" in caplog.text
        # check if augmentations were overwritten by model transform
        assert dataset.augmentations == model_transform

    @staticmethod
    def test_augmentations_as_compose() -> None:
        """Test that the Resize transform is added to the augmentations if augmentations is a Compose object."""
        dataset = DummyDataset()
        model_transform = Resize((224, 224))
        augmentations = Compose([RandomHorizontalFlip(), RandomVerticalFlip()])

        AnomalibDataModule._update_subset_augmentations(dataset, augmentations, model_transform=model_transform)  # noqa: SLF001
        assert dataset.augmentations.transforms[-1] == model_transform

    @staticmethod
    def test_augmentations_as_transform() -> None:
        """Test that the Resize transform is added to the augmentations if augmentations is a single transform."""
        dataset = DummyDataset()
        model_transform = Resize((224, 224))
        augmentations = RandomHorizontalFlip()

        AnomalibDataModule._update_subset_augmentations(dataset, augmentations, model_transform=model_transform)  # noqa: SLF001
        assert dataset.augmentations.transforms[-1] == model_transform

    @staticmethod
    def test_model_transform_as_compose() -> None:
        """Test that the Resize transform is added to the augmentations if model_transform is a Compose object."""
        dataset = DummyDataset()
        model_transform = Compose([Resize(224, 224), Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])])
        augmentations = Compose([RandomHorizontalFlip(), RandomVerticalFlip()])

        AnomalibDataModule._update_subset_augmentations(dataset, augmentations, model_transform=model_transform)  # noqa: SLF001
        assert dataset.augmentations.transforms[-1] == model_transform.transforms[0]

    @staticmethod
    def test_no_model_transforms() -> None:
        """Test that the augmentations are added but not modified if model_transform is None."""
        dataset = DummyDataset()
        augmentations = Compose([RandomHorizontalFlip(), RandomVerticalFlip()])

        AnomalibDataModule._update_subset_augmentations(dataset, augmentations, model_transform=None)  # noqa: SLF001
        assert dataset.augmentations == augmentations

    @staticmethod
    def test_no_augmentations() -> None:
        """Test that the model_transform resize is added to the augmentations if augmentations is None."""
        dataset = DummyDataset()
        model_transform = Resize((224, 224))

        AnomalibDataModule._update_subset_augmentations(dataset, augmentations=None, model_transform=model_transform)  # noqa: SLF001
        assert dataset.augmentations == model_transform
