"""Utility functions for transforms."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

from torch.utils.data import DataLoader
from torchvision.transforms.v2 import CenterCrop, Compose, Resize, Transform

from anomalib.data import AnomalibDataModule
from anomalib.data.transforms import ExportableCenterCrop


def get_stage_transform(stage: str, transforms: dict[str, Transform | None]) -> Transform | None:
    """Get the transform for a specific stage.

    Args:
        stage: The stage to get the transform for (fit, validate, test, predict).
        transforms: Dictionary mapping stage names to transforms.

    Returns:
        Transform for the specified stage, or None if not found.
    """
    stage_transforms_mapping = {
        "fit": transforms.get("train"),
        "validate": transforms.get("val"),
        "test": transforms.get("test"),
        "predict": transforms.get("test"),  # predict uses test transform
    }
    return stage_transforms_mapping.get(stage)


def get_dataloaders_transforms(dataloaders: Sequence[DataLoader]) -> dict[str, Transform]:
    """Get transforms from dataloaders.

    Args:
        dataloaders: The dataloaders to get transforms from.

    Returns:
        Dictionary mapping stages to their transforms.
    """
    transforms: dict[str, Transform] = {}
    stage_lookup = {
        "fit": "train",
        "validate": "val",
        "test": "test",
        "predict": "test",
    }

    for dataloader in dataloaders:
        if not hasattr(dataloader, "dataset") or not hasattr(dataloader.dataset, "transform"):
            continue

        for stage in stage_lookup:
            if hasattr(dataloader, f"{stage}_dataloader"):
                transforms[stage_lookup[stage]] = dataloader.dataset.transform

    return transforms


def set_datamodule_transforms(datamodule: AnomalibDataModule, transforms: dict[str, Transform | None]) -> None:
    """Set transforms to a datamodule.

    Args:
        datamodule: The datamodule to propagate transforms to.
        transforms: Dictionary mapping stages to their transforms.
    """
    for stage in ["fit", "validate", "test", "predict"]:
        transform = get_stage_transform(stage, transforms)
        if transform is not None:
            set_datamodule_stage_transform(datamodule, transform, stage)


def set_dataloaders_transforms(dataloaders: Sequence[DataLoader], transforms: dict[str, Transform | None]) -> None:
    """Set transforms to dataloaders.

    Args:
        dataloaders: The dataloaders to propagate transforms to.
        transforms: Dictionary mapping stages to their transforms.
    """
    stage_mapping = {
        "fit": "train",
        "validate": "val",
        "test": "test",
        "predict": "test",  # predict uses test transform
    }

    for loader in dataloaders:
        if not hasattr(loader, "dataset"):
            continue

        for stage in stage_mapping:
            if hasattr(loader, f"{stage}_dataloader"):
                transform = transforms.get(stage_mapping[stage])
                if transform is not None:
                    set_dataloader_transform([loader], transform)


def set_datamodule_stage_transform(datamodule: AnomalibDataModule, transform: Transform, stage: str) -> None:
    """Set a transform for a specific stage in a AnomalibDataModule.

    Args:
        datamodule: The AnomalibDataModule to set the transform for.
        transform: The transform to set.
        stage: The stage to set the transform for.

    Note:
        The stage parameter maps to dataset attributes as follows:
        - 'fit' -> 'train_data'
        - 'validate' -> 'val_data'
        - 'test' -> 'test_data'
        - 'predict' -> 'test_data'
    """
    stage_datasets = {
        "fit": "train_data",
        "validate": "val_data",
        "test": "test_data",
        "predict": "test_data",
    }

    dataset_attr = stage_datasets.get(stage)
    if dataset_attr and hasattr(datamodule, dataset_attr):
        dataset = getattr(datamodule, dataset_attr)
        if hasattr(dataset, "transform"):
            dataset.transform = transform


def set_dataloader_transform(dataloader: DataLoader | Sequence[DataLoader], transform: Transform) -> None:
    """Set a transform for a dataloader or list of dataloaders.

    Args:
        dataloader: The dataloader(s) to set the transform for.
        transform: The transform to set.
    """
    if isinstance(dataloader, DataLoader):
        if hasattr(dataloader.dataset, "transform"):
            dataloader.dataset.transform = transform
    elif isinstance(dataloader, Sequence):
        for dl in dataloader:
            set_dataloader_transform(dl, transform)
    else:
        msg = f"Unsupported dataloader type: {type(dataloader)}"
        raise TypeError(msg)


def get_exportable_transform(transform: Transform | None) -> Transform | None:
    """Get exportable transform.

    Some transforms are not supported by ONNX/OpenVINO, so we need to replace them with exportable versions.
    """
    if transform is None:
        return None
    transform = disable_antialiasing(transform)
    return convert_center_crop_transform(transform)


def disable_antialiasing(transform: Transform) -> Transform:
    """Disable antialiasing in Resize transforms.

    Resizing with antialiasing is not supported by ONNX, so we need to disable it.
    """
    if isinstance(transform, Resize):
        transform.antialias = False
    if isinstance(transform, Compose):
        for tr in transform.transforms:
            disable_antialiasing(tr)
    return transform


def convert_center_crop_transform(transform: Transform) -> Transform:
    """Convert CenterCrop to ExportableCenterCrop.

    Torchvision's CenterCrop is not supported by ONNX, so we need to replace it with our own ExportableCenterCrop.
    """
    if isinstance(transform, CenterCrop):
        transform = ExportableCenterCrop(size=transform.size)
    if isinstance(transform, Compose):
        for index in range(len(transform.transforms)):
            tr = transform.transforms[index]
            transform.transforms[index] = convert_center_crop_transform(tr)
    return transform
