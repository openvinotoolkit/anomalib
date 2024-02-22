"""Tests for Torch and OpenVINO inferencer throughput used in sweep."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path

from anomalib import TaskType
from anomalib.data.image.folder import FolderDataset
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Padim
from anomalib.pipelines.sweep.helpers import get_openvino_throughput, get_torch_throughput


def test_torch_throughput(
    project_path: Path,
    dataset_path: Path,
    ckpt_path: Callable[[str], Path],
) -> None:
    """Test get_torch_throughput from pipelines/sweep/inference.py."""
    _ckpt_path = ckpt_path("Padim")
    model = Padim.load_from_checkpoint(_ckpt_path)
    engine = Engine(default_root_dir=project_path, task=TaskType.CLASSIFICATION)
    # create Dataset from generated TestDataset
    dataset = FolderDataset(
        task=TaskType.CLASSIFICATION,
        root=dataset_path / "mvtec",
        normal_dir="dummy/test/good",
    )
    dataset.setup()
    engine.export(
        model=model,
        export_type=ExportType.TORCH,
        export_root=_ckpt_path.parent.parent,
    )

    # run procedure using torch inferencer
    get_torch_throughput(_ckpt_path.parent.parent, dataset, device="gpu")


def test_openvino_throughput(
    project_path: Path,
    dataset_path: Path,
    ckpt_path: Callable[[str], Path],
) -> None:
    """Test get_openvino_throughput from pipelines/sweep/inference.py."""
    _ckpt_path = ckpt_path("Padim")
    model = Padim.load_from_checkpoint(_ckpt_path)
    engine = Engine(default_root_dir=project_path, task=TaskType.CLASSIFICATION)
    # create Dataset from generated TestDataset
    dataset = FolderDataset(
        task=TaskType.CLASSIFICATION,
        root=dataset_path / "mvtec",
        normal_dir="dummy/test/good",
    )
    dataset.setup()
    engine.export(
        model=model,
        export_type=ExportType.OPENVINO,
        export_root=_ckpt_path.parent.parent,
    )

    # run procedure using openvino inferencer
    get_openvino_throughput(_ckpt_path.parent.parent, dataset)
