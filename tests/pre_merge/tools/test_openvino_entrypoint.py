"""Test OpenVINO inference entrypoint script."""

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
class TestOpenVINOInferenceEntrypoint:
    """This tests whether the entrypoints run without errors without quantitative measure of the outputs."""

    @pytest.fixture
    def get_functions(self):
        """Get functions from openvino_inference.py"""
        if find_spec("openvino_inference") is not None:
            from tools.inference.openvino_inference import Runner, get_parser
        else:
            raise Exception("Unable to import openvino_inference.py for testing")
        return get_parser, Runner

    def test_openvino_inference(
        self, get_functions, get_config, project_path, get_dummy_inference_image, transforms_config
    ):
        """Test openvino_inference.py"""
        get_parser, Runner = get_functions

        model = get_model(get_config("padim"))

        # export OpenVINO model
        export(
            task=TaskType.SEGMENTATION,
            transform=transforms_config,
            input_size=(100, 100),
            model=model,
            export_mode=ExportMode.OPENVINO,
            export_root=project_path,
        )

        args = get_parser().parse_args(
            [
                "--weights",
                project_path + "/weights/openvino/model.bin",
                "--input",
                get_dummy_inference_image,
                "--output",
                project_path + "/output.png",
            ]
        )
        runner = Runner(
            weights=args.weights,
            visualization_mode="simple",
            device="CPU",
            inputs=args.input,
            outputs=args.output,
            show=False,
        )
        runner.run()
