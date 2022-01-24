"""Quick sanity check on models."""

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

import os
import tempfile

import numpy as np
import pytest
from pytorch_lightning import Trainer

from anomalib.config import get_configurable_parameters, update_nncf_config
from anomalib.core.callbacks import get_callbacks
from anomalib.core.callbacks.visualizer_callback import VisualizerCallback
from anomalib.data import get_datamodule
from anomalib.models import get_model
from tests.helpers.dataset import TestDataset
from tests.helpers.model import model_load_test, setup


@pytest.mark.skipif(os.environ["NIGHTLY_BUILD"] == "TRUE", reason="Skipping the test as it is running nightly build.")
class TestModel:
    """Do a sanity check on the models."""

    @pytest.mark.parametrize(
        ["model_name", "nncf"],
        [
            ("padim", False),
            ("dfkde", False),
            ("dfm", False),
            ("stfpm", False),
            ("stfpm", True),
            ("patchcore", False),
            ("cflow", False),
        ],
    )
    @TestDataset(num_train=20, num_test=10)
    def test_model(self, model_name, nncf, category="shapes", path=""):
        """Test the models on only 1 epoch as a sanity check before merge."""
        with tempfile.TemporaryDirectory() as project_path:
            # Train test
            model, config, datamodule, trainer = setup(
                model_name,
                dataset_path=path,
                project_path=project_path,
                nncf=nncf,
                category=category,
                weight_file="",
                fast_run=True,
            )
            results = trainer.test(model=model, datamodule=datamodule)[0]

            # Test model load
            config.model.weight_file = "weights/model.ckpt"  # add model weights to the config
            model_load_test(config, datamodule, results)
