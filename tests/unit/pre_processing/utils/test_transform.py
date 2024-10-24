"""Test the pre-processing transforms utils."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms.v2 import CenterCrop, Compose, Resize, ToTensor

from anomalib.data.transforms import ExportableCenterCrop
from anomalib.pre_processing.utils.transform import (
    convert_centercrop,
    disable_antialiasing,
    get_exportable_transform,
    set_dataloader_transform,
)


def test_set_dataloader_transform() -> None:
    """Test the set_dataloader_transform function."""

    # Test with single DataLoader
    class TransformableDataset(TensorDataset):
        def __init__(self, *tensors) -> None:
            super().__init__(*tensors)
            self.transform = None

    dataset = TransformableDataset(torch.randn(10, 3, 224, 224))
    dataloader = DataLoader(dataset)
    transform = ToTensor()
    set_dataloader_transform(dataloader, transform)
    assert dataloader.dataset.transform == transform

    # Test with sequence of DataLoaders
    dataloaders = [DataLoader(TransformableDataset(torch.randn(10, 3, 224, 224))) for _ in range(3)]
    set_dataloader_transform(dataloaders, transform)
    for dl in dataloaders:
        assert dl.dataset.transform == transform

    # Test with unsupported type
    with pytest.raises(TypeError):
        set_dataloader_transform({"key": "value"}, transform)


def test_get_exportable_transform() -> None:
    """Test the get_exportable_transform function."""
    # Test with None transform
    assert get_exportable_transform(None) is None

    # Test with Resize transform
    resize = Resize((224, 224), antialias=True)
    exportable_resize = get_exportable_transform(resize)
    assert isinstance(exportable_resize, Resize)
    assert not exportable_resize.antialias

    # Test with CenterCrop transform
    center_crop = CenterCrop((224, 224))
    exportable_center_crop = get_exportable_transform(center_crop)
    assert isinstance(exportable_center_crop, ExportableCenterCrop)

    # Test with Compose transform
    compose = Compose([Resize((224, 224), antialias=True), CenterCrop((200, 200))])
    exportable_compose = get_exportable_transform(compose)
    assert isinstance(exportable_compose, Compose)
    assert isinstance(exportable_compose.transforms[0], Resize)
    assert not exportable_compose.transforms[0].antialias
    assert isinstance(exportable_compose.transforms[1], ExportableCenterCrop)


def test_disable_antialiasing() -> None:
    """Test the disable_antialiasing function."""
    # Test with Resize transform
    resize = Resize((224, 224), antialias=True)
    disabled_resize = disable_antialiasing(resize)
    assert not disabled_resize.antialias

    # Test with Compose transform
    compose = Compose([Resize((224, 224), antialias=True), ToTensor()])
    disabled_compose = disable_antialiasing(compose)
    assert not disabled_compose.transforms[0].antialias

    # Test with non-Resize transform
    to_tensor = ToTensor()
    assert disable_antialiasing(to_tensor) == to_tensor


def test_convert_centercrop() -> None:
    """Test the convert_centercrop function."""
    # Test with CenterCrop transform
    center_crop = CenterCrop((224, 224))
    converted_crop = convert_centercrop(center_crop)
    assert isinstance(converted_crop, ExportableCenterCrop)
    assert converted_crop.size == list(center_crop.size)

    # Test with Compose transform
    compose = Compose([Resize((256, 256)), CenterCrop((224, 224))])
    converted_compose = convert_centercrop(compose)
    assert isinstance(converted_compose.transforms[1], ExportableCenterCrop)

    # Test with non-CenterCrop transform
    resize = Resize((224, 224))
    assert convert_centercrop(resize) == resize
