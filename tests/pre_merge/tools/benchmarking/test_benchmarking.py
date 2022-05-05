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

import subprocess
from pathlib import Path


def check_tf_logs(model: str):
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
    config_path = "tests/pre_merge/tools/benchmarking/benchmark_params.yaml"

    command = f"python tools/benchmarking/benchmark.py --config {config_path}"
    subprocess.call(command, shell=True)
    check_tf_logs("padim")
    check_csv("padim")
