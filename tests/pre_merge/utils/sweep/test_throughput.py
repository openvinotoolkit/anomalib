"""Tests for Torch and OpenVINO inferencer throughput used in sweep."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import albumentations as A
from albumentations.pytorch import ToTensorV2

from anomalib.data import TaskType
from anomalib.data.folder import FolderDataset
from anomalib.deploy import ExportMode

from anomalib.utils.sweep.helpers import get_torch_throughput, get_openvino_throughput

from tests.helpers.dataset import TestDataset


transforms = A.Compose([A.ToFloat(max_value=255), ToTensorV2()])


@TestDataset(num_train=20, num_test=10)
def test_torch_throughput(generate_results_dir, path: str = None, category: str = "shapes"):
    """Test get_torch_throughput from utils/sweep/inference.py"""
    # generate results with torch model exported
    model_config = generate_results_dir(
        model_name="padim",
        dataset_path=path,
        task=TaskType.CLASSIFICATION,
        category=category,
        export_mode=ExportMode.TORCH,
    )

    # create Dataset from generated TestDataset
    dataset = FolderDataset(
        task=TaskType.CLASSIFICATION, transform=transforms, root=path, normal_dir=f"{category}/test/good"
    )
    dataset.setup()

    # run procedure using torch inferencer
    get_torch_throughput(model_config.project.path, dataset, device=model_config.trainer.accelerator)


@TestDataset(num_train=20, num_test=10)
def test_openvino_throughput(generate_results_dir, path: str = None, category: str = "shapes"):
    """Test get_openvino_throughput from utils/sweep/inference.py"""
    # generate results with torch model exported
    model_config = generate_results_dir(
        model_name="padim",
        dataset_path=path,
        task=TaskType.CLASSIFICATION,
        category=category,
        export_mode=ExportMode.OPENVINO,
    )

    # create Dataset from generated TestDataset
    dataset = FolderDataset(
        task=TaskType.CLASSIFICATION, transform=transforms, root=path, normal_dir=f"{category}/test/good"
    )
    dataset.setup()

    # run procedure using openvino inferencer
    get_openvino_throughput(model_config.project.path, dataset)
