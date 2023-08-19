"""Test tiled ensemble training script"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from omegaconf import OmegaConf

from tools.tiled_ensemble.train_ensemble import get_parser as get_train_parser, train
from tools.tiled_ensemble.test_ensemble import get_parser as get_test_parser, test as run_test


sys.path.append("tools")


@pytest.fixture(scope="module")
def get_mock_environment(get_ensemble_config):
    with TemporaryDirectory() as temp_dir:
        config = get_ensemble_config
        config.project.path = temp_dir

        (Path(temp_dir) / "config.yaml").write_text(OmegaConf.to_yaml(config))

        yield temp_dir


@pytest.mark.order(1)
def test_train_script(get_mock_environment):
    """Test train_ensemble.py."""
    project_path = get_mock_environment

    args = get_train_parser().parse_args(
        [
            "--model_config",
            f"{project_path}/config.yaml",
            "--ensemble_config",
            "tests/pre_merge/tools/tiled_ensemble/dummy_ens_config.yaml",
        ]
    )
    train(args)


@pytest.mark.order(2)
def test_test_script(get_mock_environment):
    """Test test_ensemble.py"""
    project_path = get_mock_environment

    args = get_test_parser().parse_args(
        [
            "--model_config",
            f"{project_path}/config.yaml",
            "--ensemble_config",
            "tests/pre_merge/tools/tiled_ensemble/dummy_ens_config.yaml",
            "--weight_folder",
            f"{project_path}/padim/mvtec/bottle/run/weights/lightning",
        ]
    )
    run_test(args)
