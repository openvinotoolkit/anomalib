"""Tests for Torch and OpenVINO inferencers."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

import pytest
import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Trainer

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.deploy import OpenVINOInferencer, TorchInferencer
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks
from tests.helpers.dataset import TestDataset, get_dataset_path
from tests.helpers.inference import MockImageLoader


def get_model_config(
    project_path: str,
    model_name: str,
    dataset_path: str,
    category: str,
    task: str = "classification",
    export_mode: Optional[str] = None,
):
    model_config = get_configurable_parameters(model_name=model_name)
    model_config.project.path = project_path
    model_config.dataset.task = task
    model_config.dataset.path = dataset_path
    model_config.dataset.category = category
    model_config.trainer.fast_dev_run = True
    model_config.trainer.max_epochs = 1
    model_config.trainer.devices = 1
    model_config.trainer.accelerator = "gpu"
    model_config.optimization.export_mode = export_mode
    return model_config


@pytest.fixture(autouse=True)
def generate_results_dir():
    with TemporaryDirectory() as project_path:

        def make(
            model_name: str,
            dataset_path: str,
            category: str,
            task: str = "classification",
            export_mode: Optional[str] = None,
        ) -> Union[DictConfig, ListConfig]:
            # then train the model
            model_config = get_model_config(
                project_path=project_path,
                model_name=model_name,
                dataset_path=dataset_path,
                category=category,
                task=task,
                export_mode=export_mode,
            )
            model = get_model(model_config)
            datamodule = get_datamodule(model_config)
            callbacks = get_callbacks(model_config)
            trainer = Trainer(**model_config.trainer, logger=False, callbacks=callbacks)
            trainer.fit(model=model, datamodule=datamodule)

            return model_config, model

        yield make


@pytest.mark.parametrize(
    "model_name, task",
    [
        ("cfa", "segmentation"),
        ("cflow", "segmentation"),
        ("dfm", "segmentation"),
        ("dfkde", "segmentation"),
        ("draem", "segmentation"),
        ("fastflow", "segmentation"),
        ("ganomaly", "segmentation"),
        ("padim", "segmentation"),
        ("patchcore", "segmentation"),
        ("reverse_distillation", "segmentation"),
        ("stfpm", "segmentation"),
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
    model_config, model = generate_results_dir(
        model_name=model_name, dataset_path=path, task=task, category=category, export_mode="torch"
    )

    # Test torch inferencer
    torch_inferencer = TorchInferencer(
        path=Path(model_config.project.path) / "weights" / "torch" / "model.pt",
        device="cpu",
    )
    torch_dataloader = MockImageLoader(model_config.dataset.image_size, total_count=1)
    with torch.no_grad():
        for image in torch_dataloader():
            prediction = torch_inferencer.predict(image)
            assert 0.0 <= prediction.pred_score <= 1.0  # confirm if predicted scores are normalized


@pytest.mark.parametrize(
    "model_name, task",
    [
        ("dfm", "classification"),
        ("draem", "segmentation"),
        ("ganomaly", "segmentation"),
        ("padim", "segmentation"),
        ("patchcore", "segmentation"),
        ("stfpm", "segmentation"),
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
    model_config, _model = generate_results_dir(
        model_name=model_name, dataset_path=path, task=task, category=category, export_mode="openvino"
    )
    export_path = Path(model_config.project.path)

    # Test OpenVINO inferencer
    openvino_inferencer = OpenVINOInferencer(
        export_path / "weights/openvino/model.xml", export_path / "weights/openvino/metadata.json"
    )
    openvino_dataloader = MockImageLoader(model_config.dataset.image_size, total_count=1)
    for image in openvino_dataloader():
        prediction = openvino_inferencer.predict(image)
        assert 0.0 <= prediction.pred_score <= 1.0  # confirm if predicted scores are normalized
