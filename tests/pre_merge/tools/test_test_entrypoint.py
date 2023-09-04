"""Test test.py entrypoint script."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from importlib.util import find_spec

import pytest

sys.path.append("tools")
from unittest.mock import patch


@pytest.mark.order(2)
class TestTestEntrypoint:
    """This tests whether the entrypoints run without errors without quantitative measure of the outputs."""

    @pytest.fixture
    def get_functions(self):
        """Get functions from test.py"""
        if find_spec("test") is not None:
            from tools.test import get_parser, test
        else:
            raise Exception("Unable to import test.py for testing")
        return get_parser, test

    def test_test(self, get_functions, get_config, project_path):
        """Test test.py"""
        get_parser, test = get_functions
        with patch("tools.test.get_configurable_parameters", side_effect=get_config):
            # Test when model key is passed
            arguments = get_parser().parse_args(
                ["--model", "padim", "--weight_file", project_path + "/weights/lightning/model.ckpt"]
            )
            test(arguments)

            # Test when weight file is incorrect
            arguments = get_parser().parse_args(["--model", "padim"])
            with pytest.raises(FileNotFoundError):
                test(arguments)

            # Test when config key is passed
            arguments = get_parser().parse_args(
                [
                    "--config",
                    "src/anomalib/models/padim/config.yaml",
                    "--weight_file",
                    project_path + "/weights/lightning/model.ckpt",
                ]
            )
            test(arguments)
