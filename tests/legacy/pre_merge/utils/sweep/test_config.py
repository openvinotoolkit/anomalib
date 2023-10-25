"""Test sweep config utils."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from omegaconf import DictConfig

from anomalib.utils.sweep.config import get_run_config, set_in_nested_config


class TestSweepConfig:
    def test_get_run_config(self):
        """Test whether the run config is returned correctly and patches the keys which have only one value."""
        dummy_config = DictConfig(
            {
                "parent1": {"child1": ["a", "b"], "child2": [1, 2]},
                "parent2": ["model1", "model2"],
                "parent3": "replacement_value",
            }
        )
        run_config = list(get_run_config(dummy_config))
        expected_value = [
            {"parent1.child1": "a", "parent1.child2": 1, "parent2": "model1", "parent3": "replacement_value"},
            {"parent1.child1": "a", "parent1.child2": 1, "parent2": "model2", "parent3": "replacement_value"},
            {"parent1.child1": "a", "parent1.child2": 2, "parent2": "model1", "parent3": "replacement_value"},
            {"parent1.child1": "a", "parent1.child2": 2, "parent2": "model2", "parent3": "replacement_value"},
            {"parent1.child1": "b", "parent1.child2": 1, "parent2": "model1", "parent3": "replacement_value"},
            {"parent1.child1": "b", "parent1.child2": 1, "parent2": "model2", "parent3": "replacement_value"},
            {"parent1.child1": "b", "parent1.child2": 2, "parent2": "model1", "parent3": "replacement_value"},
            {"parent1.child1": "b", "parent1.child2": 2, "parent2": "model2", "parent3": "replacement_value"},
        ]
        assert run_config == expected_value

    def set_in_nested_config(self):
        dummy_config = DictConfig(
            {"parent1": {"child1": ["a", "b", "c"], "child2": [1, 2, 3]}, "parent2": ["model1", "model2"]}
        )

        model_config = DictConfig(
            {
                "parent1": {
                    "child1": "e",
                    "child2": 4,
                },
                "parent3": False,
            }
        )

        for run_config in get_run_config(dummy_config):
            for param in run_config.keys():
                set_in_nested_config(model_config, param.split("."), run_config[param])
        assert model_config == {"parent1": {"child1": "a", "child2": 1}, "parent3": False, "parent2": "model1"}
