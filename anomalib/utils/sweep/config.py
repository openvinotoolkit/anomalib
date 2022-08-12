"""Utilities for modifying the configuration."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import itertools
import operator
from functools import reduce
from typing import Any, Generator, List

from omegaconf import DictConfig


def flatten_sweep_params(params_dict: DictConfig) -> DictConfig:
    """Flatten the nested parameters section of the config object.

    We need to flatten the params so that all the nested keys are concatenated into a single string.
    This is useful when
    - We need to do a cartesian product of all the combinations of the configuration for grid search.
    - Save keys as headers for csv
    - Add the config to `wandb` sweep.

    Args:
        params_dict: DictConfig: The dictionary containing the hpo parameters in the original, nested, structure.

    Returns:
        flattened version of the parameter dictionary.
    """

    def flatten_nested_dict(nested_params: DictConfig, keys: List[str], flattened_params: DictConfig):
        """Flatten nested dictionary.

        Recursive helper function that traverses the nested config object and stores the leaf nodes in a flattened
        dictionary.

        Args:
            nested_params: DictConfig: config object containing the original parameters.
            keys: List[str]: list of keys leading to the current location in the config.
            flattened_params: DictConfig: Dictionary in which the flattened parameters are stored.
        """
        for name, cfg in nested_params.items():
            if isinstance(cfg, DictConfig):
                flatten_nested_dict(cfg, keys + [str(name)], flattened_params)
            else:
                key = ".".join(keys + [str(name)])
                flattened_params[key] = cfg

    flattened_params_dict = DictConfig({})
    flatten_nested_dict(params_dict, [], flattened_params_dict)

    return flattened_params_dict


def get_run_config(params_dict: DictConfig) -> Generator[DictConfig, None, None]:
    """Yields configuration for a single run.

    Args:
        params_dict (DictConfig): Configuration for grid search.

    Example:
        >>> dummy_config = DictConfig({
            "parent1":{
                "child1": ['a', 'b', 'c'],
                "child2": [1, 2, 3]
            },
            "parent2":['model1', 'model2']
        })
        >>> for run_config in get_run_config(dummy_config):
        >>>    print(run_config)
        {'parent1.child1': 'a', 'parent1.child2': 1, 'parent2': 'model1'}
        {'parent1.child1': 'a', 'parent1.child2': 1, 'parent2': 'model2'}
        {'parent1.child1': 'a', 'parent1.child2': 2, 'parent2': 'model1'}
        ...

    Yields:
        Generator[DictConfig]: Dictionary containing flattened keys
        and values for current run.
    """
    params = flatten_sweep_params(params_dict)
    combinations = list(itertools.product(*params.values()))
    keys = params.keys()
    for combination in combinations:
        run_config = DictConfig({})
        for key, val in zip(keys, combination):
            run_config[key] = val
        yield run_config


def get_from_nested_config(config: DictConfig, keymap: List) -> Any:
    """Retrieves an item from a nested config object using a list of keys.

    Args:
        config: DictConfig: nested DictConfig object
        keymap: List[str]: list of keys corresponding to item that should be retrieved.
    """
    return reduce(operator.getitem, keymap, config)


def set_in_nested_config(config: DictConfig, keymap: List, value: Any):
    """Set an item in a nested config object using a list of keys.

    Args:
        config: DictConfig: nested DictConfig object
        keymap: List[str]: list of keys corresponding to item that should be set.
        value: Any: Value that should be assigned to the dictionary item at the specified location.

    Example:
        >>> dummy_config = DictConfig({
            "parent1":{
                "child1": ['a', 'b', 'c'],
                "child2": [1, 2, 3]
            },
            "parent2":['model1', 'model2']
        })
        >>> model_config = DictConfig({
            "parent1":{
                "child1": 'e',
                "child2": 4,
            },
            "parent3": False
        })
        >>> for run_config in get_run_config(dummy_config):
        >>>    print("Original model config", model_config)
        >>>    print("Suggested config", run_config)
        >>>    for param in run_config.keys():
        >>>        set_in_nested_config(model_config, param.split('.'), run_config[param])
        >>>    print("Replaced model config", model_config)
        >>>    break
        Original model config {'parent1': {'child1': 'e', 'child2': 4}, 'parent3': False}
        Suggested config {'parent1.child1': 'a', 'parent1.child2': 1, 'parent2': 'model1'}
        Replaced model config {'parent1': {'child1': 'a', 'child2': 1}, 'parent3': False, 'parent2': 'model1'}
    """
    get_from_nested_config(config, keymap[:-1])[keymap[-1]] = value
