"""Test torch inference entrypoint script."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import sys
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path

import pytest

from anomalib.deploy import export_to_torch
from anomalib.models import Padim
from anomalib.utils.metrics.threshold import F1AdaptiveThreshold
from anomalib.utils.types import TaskType

sys.path.append("tools/inference")


class TestTorchInferenceEntrypoint:
    """This tests whether the entrypoints run without errors without quantitative measure of the outputs."""

    @pytest.fixture
    def get_functions(self) -> tuple[Callable, Callable]:
        """Get functions from torch_inference.py."""
        if find_spec("torch_inference") is not None:
            from tools.inference.torch_inference import get_parser, infer
        else:
            raise Exception("Unable to import torch_inference.py for testing")
        return get_parser, infer

    def test_torch_inference(
        self, get_functions: tuple[Callable, Callable], project_path: Path, get_dummy_inference_image, transforms_config
    ):
        """Test torch_inference.py."""
        get_parser, infer = get_functions
        model = Padim(input_size=(100, 100))
        model.image_threshold = F1AdaptiveThreshold()
        model.pixel_threshold = F1AdaptiveThreshold()
        export_to_torch(model=model, export_path=project_path, transform=transforms_config, task=TaskType.SEGMENTATION)
        arguments = get_parser().parse_args(
            [
                "--weights",
                str(project_path) + "/weights/torch/model.pt",
                "--input",
                get_dummy_inference_image,
                "--output",
                str(project_path) + "/output.png",
            ],
        )
        infer(arguments)
