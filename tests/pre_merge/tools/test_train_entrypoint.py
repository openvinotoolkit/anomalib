"""Test train entrypoint"""


# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import sys
from importlib.util import find_spec
from unittest.mock import patch

import pytest

sys.path.append("tools")  # This is the path to the tools folder as it is not part of the anomalib package


@pytest.mark.order(1)
class TestTrainEntrypoint:
    """This tests whether the entrypoints run without errors without quantitative measure of the outputs."""

    @pytest.fixture
    def get_functions(self):
        """Get functions from train.py"""
        if find_spec("train") is not None:
            from tools.train import get_parser, train
        else:
            raise Exception("Unable to import train.py for testing")
        return get_parser, train

    def test_train(self, get_functions, get_config):
        """Test train.py."""
        with patch("tools.train.get_configurable_parameters", side_effect=get_config):
            get_parser, train = get_functions
            # Test when model key is passed
            args = get_parser().parse_args(["--model", "padim"])
            train(args)

            # Don't run fit and test the second time
            with patch("tools.train.Trainer"):
                # Test when config key is passed
                args = get_parser().parse_args(["--config", "src/anomalib/models/padim/config.yaml"])

                train(args)
                # Test when log_level key is passed
                args = get_parser().parse_args(["--log-level", "ERROR"])
                train(args)
