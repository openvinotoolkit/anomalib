"""Test benchmarking script on a subset of models and categories."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import sys

# Since tools is not part of the anomalib package, accessing benchmarking requires importlib
sys.path.append("tools/benchmarking")
from importlib.util import find_spec

if find_spec("benchmark") is not None:
    from benchmark import distribute
else:
    raise Exception("Unable to import benchmarking script for testing")


from pathlib import Path

from omegaconf import OmegaConf

from tests.helpers.dataset import get_dataset_path


def check_tb_logs(model: str):
    """check if TensorBoard logs are generated."""
    for device in ["gpu", "cpu"]:
        assert (
            len(list(Path("runs", f"{model}_{device}").glob("events.out.tfevents.*"))) > 0
        ), f"Benchmarking script didn't generate tensorboard logs for {model}"


def check_csv(model: str):
    """Check if csv files are generated"""
    for device in ["gpu", "cpu"]:
        assert Path(
            "runs", f"{model}_{device}.csv"
        ).exists(), f"Benchmarking script didn't generate csv logs for {model}"


def test_benchmarking():
    """Test if benchmarking script produces the required artifacts."""
    config_path = "tests/nightly/tools/benchmarking/benchmark_params.yaml"
    test_config = OmegaConf.load(config_path)
    test_config.grid_search.dataset["path"] = [get_dataset_path()]

    distribute(test_config)
    check_tb_logs("padim")
    check_csv("padim")
