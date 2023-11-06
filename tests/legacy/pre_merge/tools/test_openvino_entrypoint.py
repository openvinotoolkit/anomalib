"""Test OpenVINO inference entrypoint script."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import sys
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path

import pytest

from anomalib.data import TaskType
from anomalib.deploy import export_to_openvino
from anomalib.models import Padim
from anomalib.utils.metrics.threshold import F1AdaptiveThreshold

sys.path.append("tools/inference")


class TestOpenVINOInferenceEntrypoint:
    """This tests whether the entrypoints run without errors without quantitative measure of the outputs."""

    @pytest.fixture(scope="module")
    def get_functions(self) -> tuple[Callable, Callable]:
        """Get functions from openvino_inference.py."""
        if find_spec("openvino_inference") is not None:
            from tools.inference.openvino_inference import get_parser, infer
        else:
            raise Exception("Unable to import openvino_inference.py for testing")
        return get_parser, infer

    def test_openvino_inference(
        self,
        get_functions: tuple[Callable, Callable],
        project_path: Path,
        get_dummy_inference_image: str,
        transforms_config: dict,
    ) -> None:
        """Test openvino_inference.py."""
        get_parser, infer = get_functions

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

        arguments = get_parser().parse_args(
            [
                "--weights",
                str(project_path) + "/weights/openvino/model.bin",
                "--metadata",
                str(project_path) + "/weights/openvino/metadata.json",
                "--input",
                get_dummy_inference_image,
                "--output",
                str(project_path) + "/output.png",
            ],
        )
        infer(arguments)
