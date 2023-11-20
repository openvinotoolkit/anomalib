"""Tests for Torch and OpenVINO inferencers."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from anomalib.data import MVTec
from anomalib.deploy import ExportMode, OpenVINOInferencer, TorchInferencer
from anomalib.engine import Engine
from anomalib.models import Padim
from anomalib.utils.types import TaskType
from tests.legacy.helpers.inference import MockImageLoader


@pytest.mark.parametrize(
    "task",
    [
        TaskType.CLASSIFICATION,
        TaskType.DETECTION,
        TaskType.SEGMENTATION,
    ],
)
def test_torch_inference(task: TaskType, ckpt_path: Callable[[str], Path], dataset_path: Path) -> None:
    """Tests Torch inference.

    Model is not trained as this checks that the inferencers are working.

    Args:
        task (TaskType): Task type
        ckpt_path: Callable[[str], Path]: Path to trained PADIM model checkpoint.
        dataset_path (Path): Path to dummy dataset.
    """
    model = Padim()
    engine = Engine()
    export_path = ckpt_path("Padim").parent.parent
    datamodule = MVTec(root=dataset_path / "mvtec", category="dummy")
    engine.export(
        model=model,
        export_mode=ExportMode.TORCH,
        input_size=(256, 256),
        export_path=export_path,
        datamodule=datamodule,
        ckpt_path=str(ckpt_path("Padim")),
        task=task,
    )
    # Test torch inferencer
    torch_inferencer = TorchInferencer(
        path=export_path / "weights" / "torch" / "model.pt",
        device="cpu",
    )
    torch_dataloader = MockImageLoader([256, 256], total_count=1)
    with torch.no_grad():
        for image in torch_dataloader():
            prediction = torch_inferencer.predict(image)
            assert 0.0 <= prediction.pred_score <= 1.0  # confirm if predicted scores are normalized


@pytest.mark.parametrize(
    "task",
    [
        TaskType.CLASSIFICATION,
        TaskType.DETECTION,
        TaskType.SEGMENTATION,
    ],
)
def test_openvino_inference(task: TaskType, ckpt_path: Callable[[str], Path], dataset_path: Path) -> None:
    """Tests OpenVINO inference.

    Model is not trained as this checks that the inferencers are working.

    Args:
        task (TaskType): Task type
        ckpt_path: Callable[[str], Path]: Path to trained PADIM model checkpoint.
        dataset_path (Path): Path to dummy dataset.
    """
    model = Padim()
    engine = Engine()
    export_path = ckpt_path("Padim").parent.parent
    datamodule = MVTec(root=dataset_path / "mvtec", category="dummy")
    engine.export(
        model=model,
        export_mode=ExportMode.OPENVINO,
        input_size=(256, 256),
        export_path=export_path,
        datamodule=datamodule,
        ckpt_path=str(ckpt_path("Padim")),
        task=task,
    )

    # Test OpenVINO inferencer
    openvino_inferencer = OpenVINOInferencer(
        export_path / "weights/openvino/model.xml",
        export_path / "weights/openvino/metadata.json",
    )
    openvino_dataloader = MockImageLoader([256, 256], total_count=1)
    for image in openvino_dataloader():
        prediction = openvino_inferencer.predict(image)
        assert 0.0 <= prediction.pred_score <= 1.0  # confirm if predicted scores are normalized
