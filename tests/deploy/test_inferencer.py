"""Tests for Torch and OpenVINO inferencers."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union

import pytest
import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Trainer

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.deploy.inferencers import OpenVINOInferencer, TorchInferencer
from anomalib.deploy.optimize import export_convert
from anomalib.models import get_model
from tests.helpers.dataset import TestDataset, get_dataset_path
from tests.helpers.inference import MockImageLoader, get_meta_data


def get_model_config(
    model_name: str, project_path: str, dataset_path: str, category: str
) -> Union[DictConfig, ListConfig]:
    model_config = get_configurable_parameters(model_name=model_name)
    model_config.project.path = project_path
    model_config.dataset.path = dataset_path
    model_config.dataset.category = category
    model_config.trainer.max_epochs = 1
    return model_config


class TestInferencers:
    @pytest.mark.parametrize(
        "model_name",
        [
            "padim",
            "stfpm",
            "patchcore",
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
            trainer = Trainer(logger=False, **model_config.trainer)
            datamodule = get_datamodule(model_config)

            trainer.fit(model=model, datamodule=datamodule)

            model.eval()

            # Test torch inferencer
            torch_inferencer = TorchInferencer(model_config, model)
            torch_dataloader = MockImageLoader(model_config.dataset.image_size, total_count=1)
            meta_data = get_meta_data(model, model_config.dataset.image_size)
            with torch.no_grad():
                for image in torch_dataloader():
                    torch_inferencer.predict(image, superimpose=False, meta_data=meta_data)

    @pytest.mark.parametrize(
        "model_name",
        [
            "padim",
            "stfpm",
        ],
    )
    @TestDataset(num_train=20, num_test=1, path=get_dataset_path(), use_mvtec=False)
    def test_openvino_inference(self, model_name: str, category: str = "shapes", path: str = "./datasets/MVTec"):
        """Tests OpenVINO inference.
        Model is not trained as this checks that the inferencers are working.
        Args:
            model_name (str): Name of the model
        """
        with TemporaryDirectory() as project_path:
            model_config = get_model_config(
                model_name=model_name, dataset_path=path, category=category, project_path=project_path
            )
            export_path = Path(project_path)

            model = get_model(model_config)
            trainer = Trainer(logger=False, **model_config.trainer)
            datamodule = get_datamodule(model_config)
            trainer.fit(model=model, datamodule=datamodule)

            export_convert(
                model=model,
                input_size=model_config.dataset.image_size,
                onnx_path=export_path / "model.onnx",
                export_path=export_path,
            )

            # Test OpenVINO inferencer
            openvino_inferencer = OpenVINOInferencer(model_config, export_path / "model.xml")
            openvino_dataloader = MockImageLoader(model_config.dataset.image_size, total_count=1)
            meta_data = get_meta_data(model, model_config.dataset.image_size)
            for image in openvino_dataloader():
                openvino_inferencer.predict(image, superimpose=False, meta_data=meta_data)
