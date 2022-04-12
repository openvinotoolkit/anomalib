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

import tempfile

import pytest

from tests.helpers.dataset import TestDataset
from tests.helpers.model import model_load_test, setup_model_train


class TestModel:
    """Do a sanity check on the models."""

    @pytest.mark.parametrize(
        ["model_name", "nncf"],
        [
            ("padim", False),
            ("dfkde", False),
            ("dfm", False),
            ("stfpm", False),
            ("patchcore", False),
            ("cflow", False),
            ("ganomaly", False),
        ],
    )
    @TestDataset(num_train=20, num_test=10)
    def test_model(self, model_name, nncf, category="shapes", path=""):
        """Test the models on only 1 epoch as a sanity check before merge."""
        with tempfile.TemporaryDirectory() as project_path:
            # Train test
            config, datamodule, model, trainer = setup_model_train(
                model_name,
                dataset_path=path,
                project_path=project_path,
                nncf=nncf,
                category=category,
                fast_run=True,
            )
            results = trainer.test(model=model, datamodule=datamodule)[0]

            # Test model load
            model_load_test(config, datamodule, results)
