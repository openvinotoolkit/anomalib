"""Test torch inference entrypoint script."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import sys
from importlib.util import find_spec

import pytest

from anomalib.data import TaskType
from anomalib.deploy import ExportMode, export
from anomalib.models import get_model

sys.path.append("tools/inference")


@pytest.mark.order(4)
class TestTorchInferenceEntrypoint:
    """This tests whether the entrypoints run without errors without quantitative measure of the outputs."""

    @pytest.fixture
    def get_functions(self):
        """Get functions from torch_inference.py"""
        if find_spec("torch_inference") is not None:
            from tools.inference.torch_inference import get_parser, infer
        else:
            raise Exception("Unable to import torch_inference.py for testing")
        return get_parser, infer

    def test_torch_inference(
        self, get_functions, get_config, project_path, get_dummy_inference_image, transforms_config
    ):
        """Test torch_inference.py"""
        get_parser, infer = get_functions
        model = get_model(get_config("padim"))
        export(
            task=TaskType.SEGMENTATION,
            transform=transforms_config,
            input_size=(100, 100),
            model=model,
            export_mode=ExportMode.TORCH,
            export_root=project_path,
        )
        arguments = get_parser().parse_args(
            [
                "--weights",
                project_path + "/weights/torch/model.pt",
                "--input",
                get_dummy_inference_image,
                "--output",
                project_path + "/output.png",
            ]
        )
        infer(arguments)
