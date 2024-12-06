"""Test Gradio inference entrypoint script."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path

import pytest

from anomalib import TaskType
from anomalib.deploy import OpenVINOInferencer, TorchInferencer
from anomalib.models import Padim

sys.path.append("tools/inference")


class TestGradioInferenceEntrypoint:
    """This tests whether the entrypoints run without errors without quantitative measure of the outputs.

    Note: This does not launch the gradio server. It only checks if the right inferencer is called.
    """

    @pytest.fixture()
    @staticmethod
    def get_functions() -> tuple[Callable, Callable]:
        """Get functions from Gradio_inference.py."""
        if find_spec("gradio_inference") is not None:
            from tools.inference.gradio_inference import get_inferencer, get_parser
        else:
            msg = "Unable to import gradio_inference.py for testing"
            raise ImportError(msg)
        return get_parser, get_inferencer

    @staticmethod
    def test_torch_inference(
        get_functions: tuple[Callable, Callable],
        ckpt_path: Callable[[str], Path],
    ) -> None:
        """Test gradio_inference.py."""
        _ckpt_path = ckpt_path("Padim")
        parser, inferencer = get_functions
        model = Padim.load_from_checkpoint(_ckpt_path)

        # export torch model
        model.to_torch(
            export_root=_ckpt_path.parent.parent.parent,
            task=TaskType.SEGMENTATION,
        )

        arguments = parser().parse_args(
            [
                "--weights",
                str(_ckpt_path.parent.parent) + "/torch/model.pt",
            ],
        )
        assert isinstance(inferencer(arguments.weights, arguments.metadata), TorchInferencer)

    @staticmethod
    def test_openvino_inference(
        get_functions: tuple[Callable, Callable],
        ckpt_path: Callable[[str], Path],
    ) -> None:
        """Test gradio_inference.py."""
        _ckpt_path = ckpt_path("Padim")
        parser, inferencer = get_functions
        model = Padim.load_from_checkpoint(_ckpt_path)

        # export OpenVINO model
        model.to_openvino(
            export_root=_ckpt_path.parent.parent.parent,
            ov_args={},
            task=TaskType.SEGMENTATION,
        )

        arguments = parser().parse_args(
            [
                "--weights",
                str(_ckpt_path.parent.parent) + "/openvino/model.bin",
                "--metadata",
                str(_ckpt_path.parent.parent) + "/openvino/metadata.json",
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
