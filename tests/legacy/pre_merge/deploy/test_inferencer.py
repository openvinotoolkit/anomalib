"""Tests for Torch and OpenVINO inferencers."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch
from tests.legacy.helpers.dataset import TestDataset, get_dataset_path
from tests.legacy.helpers.inference import MockImageLoader

from anomalib.data import MVTec
from anomalib.deploy import ExportMode, OpenVINOInferencer, TorchInferencer
from anomalib.engine import Engine
from anomalib.models import get_model
from anomalib.utils.types import TaskType


@pytest.fixture(autouse=True)
def generate_results_dir():
    with TemporaryDirectory() as project_path:

        def make(
            model_name: str,
            dataset_path: str,
            category: str,
            export_mode: ExportMode,
            task: TaskType = TaskType.CLASSIFICATION,
        ) -> Engine:
            model = get_model(model_name)
            datamodule = MVTec(root=dataset_path, category=category)
            engine = Engine(
                fast_dev_run=True,
                max_epochs=1,
                devices=1,
                accelerator="gpu",
                default_root_dir=project_path,
                logger=False,
                task=task,
            )
            engine.fit(model=model, datamodule=datamodule)
            engine.export(model=model, datamodule=datamodule, task=task, export_mode=export_mode, input_size=(256, 256))

            return engine

        yield make


@pytest.mark.parametrize(
    "model_name, task",
    [
        # ("cfa", "segmentation"),
        # ("cflow", "segmentation"),
        # ("dfm", "segmentation"),
        # ("dfkde", "segmentation"),
        # ("draem", "segmentation"),
        # ("fastflow", "segmentation"),
        # ("ganomaly", "segmentation"),
        ("padim", "segmentation"),
        # ("patchcore", "segmentation"),
        # ("reverse_distillation", "segmentation"),
        # ("stfpm", "segmentation"),
        # also test different task types for a single model
        ("padim", "classification"),
        ("padim", "detection"),
    ],
)
@TestDataset(num_train=20, num_test=1, path=get_dataset_path(), use_mvtec=False)
def test_torch_inference(
    model_name: str, generate_results_dir, task: str, category: str = "shapes", path: str = "./datasets/MVTec"
):
    """Tests Torch inference.
    Model is not trained as this checks that the inferencers are working.
    Args:
        model_name (str): Name of the model
    """
    engine = generate_results_dir(
        model_name=model_name, dataset_path=path, task=task, category=category, export_mode=ExportMode.TORCH
    )

    # Test torch inferencer
    torch_inferencer = TorchInferencer(
        path=Path(engine.trainer.default_root_dir) / "weights" / "torch" / "model.pt",
        device="cpu",
    )
    torch_dataloader = MockImageLoader([256, 256], total_count=1)
    with torch.no_grad():
        for image in torch_dataloader():
            prediction = torch_inferencer.predict(image)
            assert 0.0 <= prediction.pred_score <= 1.0  # confirm if predicted scores are normalized


@pytest.mark.parametrize(
    "model_name, task",
    [
        # ("dfm", "classification"),
        # ("draem", "segmentation"),
        # ("ganomaly", "segmentation"),
        ("padim", "segmentation"),
        # ("patchcore", "segmentation"),
        # ("stfpm", "segmentation"),
        # task types
        ("padim", "classification"),
        ("padim", "detection"),
    ],
)
@TestDataset(num_train=20, num_test=1, path=get_dataset_path(), use_mvtec=False)
def test_openvino_inference(
    model_name: str, generate_results_dir, task: str, category: str = "shapes", path: str = "./datasets/MVTec"
):
    """Tests OpenVINO inference.
    Model is not trained as this checks that the inferencers are working.
    Args:
        model_name (str): Name of the model
    """
    engine = generate_results_dir(
        model_name=model_name, dataset_path=path, task=task, category=category, export_mode=ExportMode.OPENVINO
    )
    export_path = Path(engine.trainer.default_root_dir)

    # Test OpenVINO inferencer
    openvino_inferencer = OpenVINOInferencer(
        export_path / "weights/openvino/model.xml", export_path / "weights/openvino/metadata.json"
    )
    openvino_dataloader = MockImageLoader([256, 256], total_count=1)
    for image in openvino_dataloader():
        prediction = openvino_inferencer.predict(image)
        assert 0.0 <= prediction.pred_score <= 1.0  # confirm if predicted scores are normalized
