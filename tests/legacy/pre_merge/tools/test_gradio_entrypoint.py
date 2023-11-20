"""Test Gradio inference entrypoint script."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Callable

import pytest

from anomalib.deploy import OpenVINOInferencer, TorchInferencer, export_to_openvino, export_to_torch
from anomalib.models import Padim
from anomalib.utils.metrics.threshold import F1AdaptiveThreshold
from anomalib.utils.types import TaskType

sys.path.append("tools/inference")


class TestGradioInferenceEntrypoint:
    """This tests whether the entrypoints run without errors without quantitative measure of the outputs.

    Note: This does not launch the gradio server. It only checks if the right inferencer is called.
    """

    @pytest.fixture
    def get_functions(self) -> tuple[Callable, Callable]:
        """Get functions from Gradio_inference.py."""
        if find_spec("gradio_inference") is not None:
            from tools.inference.gradio_inference import get_inferencer, get_parser
        else:
            msg = "Unable to import gradio_inference.py for testing"
            raise Exception(msg)
        return get_parser, get_inferencer

    def test_torch_inference(
        self, get_functions: tuple[Callable, Callable], project_path: Path, transforms_config: dict
    ) -> None:
        """Test gradio_inference.py."""
        parser, inferencer = get_functions
        model = Padim(input_size=(100, 100))
        model.image_threshold = F1AdaptiveThreshold()
        model.pixel_threshold = F1AdaptiveThreshold()

        # export torch model
        export_to_torch(model=model, export_path=project_path, transform=transforms_config, task=TaskType.SEGMENTATION)

        arguments = parser().parse_args(
            [
                "--weights",
                str(project_path) + "/weights/torch/model.pt",
            ],
        )
        assert isinstance(inferencer(arguments.weights, arguments.metadata), TorchInferencer)

    def test_openvino_inference(
        self, get_functions: tuple[Callable, Callable], project_path: Path, transforms_config: dict
    ) -> None:
        """Test gradio_inference.py."""
        parser, inferencer = get_functions
        model = Padim(input_size=(100, 100))
        model.image_threshold = F1AdaptiveThreshold()
        model.pixel_threshold = F1AdaptiveThreshold()

        # export OpenVINO model
        export_to_openvino(
            export_path=project_path,
            model=model,
            input_size=(100, 100),
            transform=transforms_config,
            mo_args={},
            task=TaskType.SEGMENTATION,
        )

        arguments = parser().parse_args(
            [
                "--weights",
                str(project_path) + "/weights/openvino/model.bin",
                "--metadata",
                str(project_path) + "/weights/openvino/metadata.json",
            ],
        )
        assert isinstance(inferencer(arguments.weights, arguments.metadata), OpenVINOInferencer)

        # test error is raised when metadata is not provided to openvino model
        with pytest.raises(ValueError):
            arguments = parser().parse_args(
                [
                    "--weights",
                    str(project_path) + "/weights/openvino/model.bin",
                ]
            )
            inferencer(arguments.weights, arguments.metadata)
