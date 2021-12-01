"""Test Models."""

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

import random
import tempfile
from functools import wraps
from typing import Generator

import numpy as np
import pytest
from pytorch_lightning import Trainer

from anomalib.config import get_configurable_parameters, update_nncf_config
from anomalib.core.callbacks import get_callbacks
from anomalib.core.callbacks.visualizer_callback import VisualizerCallback
from anomalib.data import get_datamodule
from anomalib.models import get_model
from tests.helpers.dataset import TestDataset, get_dataset_path


@pytest.fixture(autouse=True)
def category() -> str:
    """PyTest fixture to randomly return an MVTec category.

    Returns:
        str: Random MVTec category to train/test.
    """
    categories = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]

    category = random.choice(categories)
    return category


class AddDFMScores:
    """Function wrapper for checking both scores of DFM."""

    def __call__(self, func):
        @wraps(func)
        def inner(*args, **kwds):
            if kwds["model_name"] == "dfm":
                for score in ["fre", "nll"]:
                    func(*args, score_type=score, **kwds)
            else:
                func(*args, **kwds)

        return inner


class TestModel:
    """Test model."""

    def _setup(self, model_name, use_mvtec, dataset_path, project_path, nncf, category, score_type=None):
        config = get_configurable_parameters(model_name=model_name)
        if score_type is not None:
            config.model.score_type = score_type
        config.project.seed = 1234
        config.dataset.category = category
        config.dataset.path = dataset_path
        config.model.weight_file = "weights/model.ckpt"  # add model weights to the config

        if not use_mvtec:
            config.dataset.category = "shapes"

        if nncf:
            config.optimization.nncf.apply = True
            config = update_nncf_config(config)
            config.init_weights = None

        # reassign project path as config is updated in `update_config_for_nncf`
        config.project.path = project_path

        datamodule = get_datamodule(config)
        model = get_model(config)

        callbacks = get_callbacks(config)

        for index, callback in enumerate(callbacks):
            if isinstance(callback, VisualizerCallback):
                callbacks.pop(index)
                break

        # Train the model.
        trainer = Trainer(callbacks=callbacks, **config.trainer)
        trainer.fit(model=model, datamodule=datamodule)
        return model, config, datamodule, trainer

    def _test_metrics(self, trainer, config, model, datamodule):
        """Tests the model metrics but also acts as a setup."""

        results = trainer.test(model=model, datamodule=datamodule)[0]

        assert results["image_AUROC"] >= 0.6

        if config.dataset.task == "segmentation":
            assert results["pixel_AUROC"] >= 0.6
        return results

    def _test_model_load(self, config, datamodule, results):
        loaded_model = get_model(config)  # get new model

        callbacks = get_callbacks(config)

        for index, callback in enumerate(callbacks):
            # Remove visualizer callback as saving results takes time
            if isinstance(callback, VisualizerCallback):
                callbacks.pop(index)
                break

        # create new trainer object with LoadModel callback (assumes it is present)
        trainer = Trainer(callbacks=callbacks, **config.trainer)
        # Assumes the new model has LoadModel callback and the old one had ModelCheckpoint callback
        new_results = trainer.test(model=loaded_model, datamodule=datamodule)[0]
        assert np.isclose(
            results["image_AUROC"], new_results["image_AUROC"]
        ), "Loaded model does not yield close performance results"
        if config.dataset.task == "segmentation":
            assert np.isclose(
                results["pixel_AUROC"], new_results["pixel_AUROC"]
            ), "Loaded model does not yield close performance results"

    @pytest.mark.parametrize(
        ["model_name", "nncf"],
        [
            ("padim", False),
            ("dfkde", False),
            ("dfm", False),
            ("stfpm", False),
            ("stfpm", True),
            ("patchcore", False),
        ],
    )
    @pytest.mark.flaky(max_runs=3)
    @TestDataset(num_train=200, num_test=10, path=get_dataset_path(), use_mvtec=True)
    @AddDFMScores()
    def test_model(self, category, model_name, nncf, use_mvtec=True, path="./datasets/MVTec", score_type=None):
        """Driver for all the tests in the class."""
        with tempfile.TemporaryDirectory() as project_path:
            model, config, datamodule, trainer = self._setup(
                model_name=model_name,
                use_mvtec=use_mvtec,
                dataset_path=path,
                nncf=nncf,
                project_path=project_path,
                category=category,
                score_type=score_type,
            )

            # test model metrics
            results = self._test_metrics(trainer=trainer, config=config, model=model, datamodule=datamodule)

            # test model load
            self._test_model_load(config=config, datamodule=datamodule, results=results)
