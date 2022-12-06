"""Tests for Torch and OpenVINO inferencers."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import pytest
import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Trainer

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.deploy import OpenVINOInferencer, TorchInferencer, export
from anomalib.deploy.export import ExportMode
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks, instantiate_callbacks
from anomalib.utils.cli.helpers import configure_optimizer
from tests.helpers.dataset import TestDataset, get_dataset_path
from tests.helpers.inference import MockImageLoader


def get_model_config(
    model_name: str, project_path: str, dataset_path: str, category: str
) -> Union[DictConfig, ListConfig]:
    model_config = get_configurable_parameters(model_name=model_name)
    model_config.trainer.default_root_dir = project_path
    model_config.data.init_args.root = dataset_path
    model_config.data.init_args.category = category
    model_config.trainer.max_epochs = 1
    model_config.trainer.devices = 1
    model_config.trainer.accelerator = "gpu"
    return model_config


class TestInferencers:
    @pytest.mark.parametrize(
        "model_name",
        [
            "cflow",
            "dfm",
            "dfkde",
            "draem",
            "fastflow",
            "ganomaly",
            "padim",
            "patchcore",
            "reverse_distillation",
            "stfpm",
        ],
    )
    @TestDataset(num_train=20, num_test=1, path=get_dataset_path(), use_mvtec=False)
    def test_torch_inference(self, model_name: str, category: str = "shapes", path: str = "./datasets/MVTec"):
        """Tests Torch inference.
        Model is not trained as this checks that the inferencers are working.
        Args:
            model_name (str): Name of the model
        """
        with TemporaryDirectory() as project_path:
            model_config = get_model_config(
                model_name=model_name, dataset_path=path, category=category, project_path=project_path
            )

            model = get_model(model_config)
            datamodule = get_datamodule(model_config)
            callbacks = get_callbacks(model_config)
            callbacks = instantiate_callbacks(callbacks)
            # Remove callbacks from trainer as it is passed separately
            if "callbacks" in model_config.trainer:
                model_config.trainer.pop("callbacks")

            configure_optimizer(model, model_config)
            trainer = Trainer(**model_config.trainer, logger=False, callbacks=callbacks)

            trainer.fit(model=model, datamodule=datamodule)

            model.eval()

            # Test torch inferencer
            torch_inferencer = TorchInferencer(model_config, model, device="cpu")
            torch_dataloader = MockImageLoader(model_config.data.init_args.image_size, total_count=1)
            with torch.no_grad():
                for image in torch_dataloader():
                    prediction = torch_inferencer.predict(image)
                    assert 0.0 <= prediction.pred_score <= 1.0  # confirm if predicted scores are normalized

    @pytest.mark.parametrize(
        "model_name",
        ["dfm", "draem", "ganomaly", "padim", "patchcore", "stfpm"],
    )
    @TestDataset(num_train=20, num_test=1, path=get_dataset_path(), use_mvtec=False)
    def test_openvino_inference(self, model_name: str, category: str = "shapes", path: str = "./datasets/MVTec"):
        """Tests OpenVINO inference.
        Model is not trained as this checks that the inferencers are working.image
        Args:
            model_name (str): Name of the model
        """
        with TemporaryDirectory() as project_path:
            model_config = get_model_config(
                model_name=model_name, dataset_path=path, category=category, project_path=project_path
            )
            export_path = Path(project_path)

            model = get_model(model_config)
            datamodule = get_datamodule(model_config)
            callbacks = get_callbacks(model_config)
            callbacks = instantiate_callbacks(callbacks)
            # Remove callbacks from trainer as it is passed separately
            if "callbacks" in model_config.trainer:
                model_config.trainer.pop("callbacks")
            configure_optimizer(model, model_config)
            trainer = Trainer(**model_config.trainer, logger=False, callbacks=callbacks)

            trainer.fit(model=model, datamodule=datamodule)

            export(
                model=model,
                input_size=model_config.data.init_args.image_size,
                export_root=export_path,
                export_mode=ExportMode.OPENVINO,
            )

            # Test OpenVINO inferencer
            openvino_inferencer = OpenVINOInferencer(
                model_config, export_path / "openvino/model.xml", export_path / "openvino/meta_data.json"
            )
            openvino_dataloader = MockImageLoader(model_config.data.init_args.image_size, total_count=1)
            for image in openvino_dataloader():
                prediction = openvino_inferencer.predict(image)
                assert 0.0 <= prediction.pred_score <= 1.0  # confirm if predicted scores are normalized
