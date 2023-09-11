"""Test lightning inference entrypoint script."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import sys
from importlib.util import find_spec

import pytest

sys.path.append("tools/inference")
from unittest.mock import patch


@pytest.mark.order(3)
class TestLightningInferenceEntrypoint:
    """This tests whether the entrypoints run without errors without quantitative measure of the outputs."""

    @pytest.fixture
    def get_functions(self):
        """Get functions from lightning_inference.py"""
        if find_spec("lightning_inference") is not None:
            from tools.inference.lightning_inference import get_parser, infer
        else:
            raise Exception("Unable to import lightning_inference.py for testing")
        return get_parser, infer

    def test_lightning_inference(self, get_functions, get_config, project_path, get_dummy_inference_image):
        """Test lightning_inferenc.py"""
        get_parser, infer = get_functions
        with patch("tools.inference.lightning_inference.get_configurable_parameters", side_effect=get_config):
            arguments = get_parser().parse_args(
                [
                    "--config",
                    "src/anomalib/models/padim/config.yaml",
                    "--weights",
                    project_path + "/weights/lightning/model.ckpt",
                    "--input",
                    get_dummy_inference_image,
                    "--output",
                    project_path + "/output",
                ]
            )
            infer(arguments)
