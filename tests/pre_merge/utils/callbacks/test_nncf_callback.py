"""Test NNCF Callback."""


# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

from tests.helpers.dataset import TestDataset
from tests.helpers.model import setup_model_train


class TestNNCFCallback:
    """Test NNCF Callback."""

    @TestDataset(num_train=20, num_test=10)
    def test_nncf_callback(self, category="shapes", path=""):
        """Train PaDiM model and check if it can be exported to NNCF.

        The idea is to check if the callback is working and if the model can be exported to NNCF.
        """
        with tempfile.TemporaryDirectory() as project_path:
            config, _, _, _ = setup_model_train(
                "padim",
                dataset_path=path,
                project_path=project_path,
                nncf=True,
                category=category,
                fast_run=True,
            )
            exported_model_path = Path(config.project.path) / "compressed" / "model_nncf.onnx"
            assert exported_model_path.exists(), "NNCF model was not exported."
