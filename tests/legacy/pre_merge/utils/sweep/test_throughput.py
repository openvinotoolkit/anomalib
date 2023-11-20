"""Tests for Torch and OpenVINO inferencer throughput used in sweep."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path

import albumentations as A  # noqa: N812
import pytest
from albumentations.pytorch import ToTensorV2
from tests.legacy.helpers.dataset import TestDataset

from anomalib.data.image.folder import FolderDataset
from anomalib.deploy import ExportMode
from anomalib.pipelines.sweep.helpers import get_openvino_throughput, get_torch_throughput
from anomalib.utils.types import TaskType

transforms = A.Compose([A.ToFloat(max_value=255), ToTensorV2()])


@pytest.mark.xfail()
@TestDataset(num_train=20, num_test=10)
def test_torch_throughput(project_path: Path, path: str | None = None, category: str = "shapes") -> None:
    """Test get_torch_throughput from utils/sweep/inference.py."""
    # generate results with torch model exported
    engine = generate_results_dir(
        model_name="padim",
        dataset_path=path,
        category=category,
        export_mode=ExportMode.TORCH,
        task=TaskType.CLASSIFICATION,
    )

    # create Dataset from generated TestDataset
    dataset = FolderDataset(
        task=TaskType.CLASSIFICATION,
        transform=transforms,
        root=path,
        normal_dir=f"{category}/test/good",
    )
    dataset.setup()

    # run procedure using torch inferencer
    get_torch_throughput(project_path, dataset, device="gpu")


@pytest.mark.xfail()
@TestDataset(num_train=20, num_test=10)
def test_openvino_throughput(generate_results_dir: Callable, path: str | None = None, category: str = "shapes") -> None:
    """Test get_openvino_throughput from utils/sweep/inference.py."""
    # generate results with torch model exported
    engine = generate_results_dir(
        export_mode=ExportMode.OPENVINO,
    )

    # create Dataset from generated TestDataset
    dataset = FolderDataset(
        task=TaskType.CLASSIFICATION,
        transform=transforms,
        root=path,
        normal_dir=f"{category}/test/good",
    )
    dataset.setup()

    # run procedure using openvino inferencer
    get_openvino_throughput(engine.trainer.default_root_dir, dataset)
