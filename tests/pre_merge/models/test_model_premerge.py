"""Quick sanity check on models."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tempfile

import pytest

from tests.helpers.dataset import TestDataset
from tests.helpers.model import model_load_test, setup_model_train


class TestModel:
    """Do a sanity check on the models."""

    @pytest.mark.parametrize(
        ["model_name", "nncf"],
        [
            ("cflow", False),
            ("dfkde", False),
            ("dfm", False),
            ("draem", False),
            ("fastflow", False),
            ("ganomaly", False),
            ("padim", False),
            ("patchcore", False),
            ("reverse_distillation", False),
            ("stfpm", False),
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
