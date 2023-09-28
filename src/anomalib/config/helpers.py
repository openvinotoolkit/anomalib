"""Utilities to help serialize/deserialize the config."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

from jsonargparse import Namespace
from omegaconf import DictConfig, ListConfig, OmegaConf


def to_yaml(config: Namespace | ListConfig | DictConfig) -> str:
    """Converts the config to a yaml string

    Args:
        config (Namespace | ListConfig | DictConfig): Config

    Returns:
        str: YAML string
    """
    _config = {}
    if isinstance(config, Namespace):
        for key, value in config.items():
            if key == "config":
                continue
            if isinstance(value, Path):
                value = str(value)
            _config[key] = value
        _config = flattened_to_nested(_config)
    else:
        _config = config

    return OmegaConf.to_yaml(_config)


def flattened_to_nested(flat_dict: dict) -> dict:
    """Converts a flattened dict to a nested dict.

    {
        "key1.a": 1
        "key1.b: 2
    }

    to

    {
        "key1":
        {
            "a": 1,
            "b: 2
        }
    }

    Args:
        flat_dict (dict): Flattened dict

    Returns:
        dict: Nested dict
    """
    nested_dict: dict[str, Any] = {}
    for key, value in flat_dict.items():
        keys = key.split(".")
        current_level = nested_dict

        for k in keys[:-1]:
            if k not in current_level:
                current_level[k] = {}
            current_level = current_level[k]

        current_level[keys[-1]] = value

    return nested_dict
