"""Test lightning inference entrypoint script."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path

import pytest

sys.path.append("tools/inference")


class TestLightningInferenceEntrypoint:
    """This tests whether the entrypoints run without errors without quantitative measure of the outputs."""

    @pytest.fixture()
    @staticmethod
    def get_functions() -> tuple[Callable, Callable]:
        """Get functions from lightning_inference.py."""
        if find_spec("lightning_inference") is not None:
            from tools.inference.lightning_inference import get_parser, infer
        else:
            msg = "Unable to import lightning_inference.py for testing"
            raise ImportError(msg)
        return get_parser, infer

    @staticmethod
    def test_lightning_inference(
        get_functions: tuple[Callable, Callable],
        project_path: Path,
        get_dummy_inference_image: str,
        ckpt_path: Callable[[str], Path],
    ) -> None:
        """Test lightning_inference.py."""
        get_parser, infer = get_functions
        arguments = get_parser().parse_args(
            [
                "--model",
                "anomalib.models.Padim",
                "--ckpt_path",
                str(ckpt_path("Padim")),
                "--data.path",
                get_dummy_inference_image,
                "--output",
                str(project_path) + "/output",
            ],
        )
        infer(arguments)
