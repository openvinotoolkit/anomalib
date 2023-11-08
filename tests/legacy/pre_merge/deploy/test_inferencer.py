"""Tests for Torch and OpenVINO inferencers."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

import pytest
import torch
from lightning.pytorch import Trainer
from omegaconf import DictConfig, ListConfig
from tests.legacy.helpers.dataset import TestDataset, get_dataset_path
from tests.legacy.helpers.inference import MockImageLoader

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.deploy import OpenVINOInferencer, TorchInferencer
from anomalib.engine import Engine
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks


def get_model_config(
    project_path: str,
    model_name: str,
    dataset_path: str,
    category: str,
    task: str = "classification",
    export_mode: Optional[str] = None,
):
    model_config = get_configurable_parameters(model_name=model_name)
    model_config.trainer.default_root_dir = project_path
    model_config.data.init_args.task = task
    model_config.data.init_args.root = dataset_path
    model_config.data.init_args.category = category
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
            engine = Engine(
                **model_config.trainer,
                logger=False,
                callbacks=callbacks,
                normalization=model_config.normalization.normalization_method,
                threshold=model_config.metrics.threshold,
                task=model_config.task,
                image_metrics=model_config.metrics.get("image", None),
                pixel_metrics=model_config.metrics.get("pixel", None),
                visualization=model_config.visualization
            )
            engine.fit(model=model, datamodule=datamodule)

            return model_config, model

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
    model_config, model = generate_results_dir(
        model_name=model_name, dataset_path=path, task=task, category=category, export_mode="torch"
    )

    # Test torch inferencer
    torch_inferencer = TorchInferencer(
        path=Path(model_config.trainer.default_root_dir) / "weights" / "torch" / "model.pt",
        device="cpu",
    )
    torch_dataloader = MockImageLoader(model_config.data.init_args.image_size, total_count=1)
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
    model_config, _model = generate_results_dir(
        model_name=model_name, dataset_path=path, task=task, category=category, export_mode="openvino"
    )
    export_path = Path(model_config.trainer.default_root_dir)

    # Test OpenVINO inferencer
    openvino_inferencer = OpenVINOInferencer(
        export_path / "weights/openvino/model.xml", export_path / "weights/openvino/metadata.json"
    )
    openvino_dataloader = MockImageLoader(model_config.data.init_args.image_size, total_count=1)
    for image in openvino_dataloader():
        prediction = openvino_inferencer.predict(image)
        assert 0.0 <= prediction.pred_score <= 1.0  # confirm if predicted scores are normalized
