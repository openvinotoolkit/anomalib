"""Tests for benchmarking configuration utils."""

# Copyright (C) 2020 Intel Corporation
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

from omegaconf import DictConfig

from anomalib.utils.sweep.config import (
    flatten_sweep_params,
    get_run_config,
    set_in_nested_config,
)


class TestSweepConfig:
    def test_flatten_params(self):
        # simulate grid search config
        dummy_config = DictConfig(
            {"parent1": {"child1": ["a", "b", "c"], "child2": [1, 2, 3]}, "parent2": ["model1", "model2"]}
        )
        dummy_config = flatten_sweep_params(dummy_config)
        assert dummy_config == {
            "parent1.child1": ["a", "b", "c"],
            "parent1.child2": [1, 2, 3],
            "parent2": ["model1", "model2"],
        }

    def test_get_run_config(self):
        # simulate model config
        model_config = DictConfig(
            {
                "parent1": {
                    "child1": "e",
                    "child2": 4,
                },
                "parent3": False,
            }
        )
        # simulate grid search config
        dummy_config = DictConfig({"parent1": {"child1": ["a"], "child2": [1, 2]}, "parent2": ["model1"]})

        config_iterator = get_run_config(dummy_config)
        # First iteration
        run_config = next(config_iterator)
        assert run_config == {"parent1.child1": "a", "parent1.child2": 1, "parent2": "model1"}
        for param in run_config.keys():
            set_in_nested_config(model_config, param.split("."), run_config[param])
        assert model_config == {"parent1": {"child1": "a", "child2": 1}, "parent3": False, "parent2": "model1"}

        # Second iteration
        run_config = next(config_iterator)
        assert run_config == {"parent1.child1": "a", "parent1.child2": 2, "parent2": "model1"}
        for param in run_config.keys():
            set_in_nested_config(model_config, param.split("."), run_config[param])
        assert model_config == {"parent1": {"child1": "a", "child2": 2}, "parent3": False, "parent2": "model1"}
