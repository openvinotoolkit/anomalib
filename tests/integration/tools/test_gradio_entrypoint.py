"""Test Gradio inference entrypoint script."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import sys
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path

import pytest

from anomalib.deploy import OpenVINOInferencer, TorchInferencer, export_to_openvino, export_to_torch
from anomalib.models import Padim
from anomalib.utils.types import TaskType

sys.path.append("tools/inference")


class TestGradioInferenceEntrypoint:
    """This tests whether the entrypoints run without errors without quantitative measure of the outputs.

    Note: This does not launch the gradio server. It only checks if the right inferencer is called.
    """

    @pytest.fixture()
    def get_functions(self) -> tuple[Callable, Callable]:
        """Get functions from Gradio_inference.py."""
        if find_spec("gradio_inference") is not None:
            from tools.inference.gradio_inference import get_inferencer, get_parser
        else:
            msg = "Unable to import gradio_inference.py for testing"
            raise ImportError(msg)
        return get_parser, get_inferencer

    def test_torch_inference(
        self,
        get_functions: tuple[Callable, Callable],
        ckpt_path: Callable[[str], Path],
        transforms_config: dict,
    ) -> None:
        """Test gradio_inference.py."""
        _ckpt_path = ckpt_path("Padim")
        parser, inferencer = get_functions
        model = Padim.load_from_checkpoint(_ckpt_path)

        # export torch model
        export_to_torch(
            model=model,
            export_root=_ckpt_path.parent.parent,
            transform=transforms_config,
            task=TaskType.SEGMENTATION,
        )

        arguments = parser().parse_args(
            [
                "--weights",
                str(_ckpt_path.parent) + "/torch/model.pt",
            ],
        )
        assert isinstance(inferencer(arguments.weights, arguments.metadata), TorchInferencer)

    def test_openvino_inference(
        self,
        get_functions: tuple[Callable, Callable],
        ckpt_path: Callable[[str], Path],
        transforms_config: dict,
    ) -> None:
        """Test gradio_inference.py."""
        _ckpt_path = ckpt_path("Padim")
        parser, inferencer = get_functions
        model = Padim.load_from_checkpoint(_ckpt_path)

        # export OpenVINO model
        export_to_openvino(
            export_root=_ckpt_path.parent.parent,
            model=model,
            input_size=(256, 256),
            transform=transforms_config,
            ov_args={},
            task=TaskType.SEGMENTATION,
        )

        arguments = parser().parse_args(
            [
                "--weights",
                str(_ckpt_path.parent) + "/openvino/model.bin",
                "--metadata",
                str(_ckpt_path.parent) + "/openvino/metadata.json",
            ],
        )
        assert isinstance(inferencer(arguments.weights, arguments.metadata), OpenVINOInferencer)

        # test error is raised when metadata is not provided to openvino model
        arguments = parser().parse_args(
            [
                "--weights",
                str(_ckpt_path) + "/openvino/model.bin",
            ],
        )
        with pytest.raises(ValueError):  # noqa: PT011
            inferencer(arguments.weights, arguments.metadata)
