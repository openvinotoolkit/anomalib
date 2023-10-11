"""Utilities to help serialize/deserialize the config."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from jsonargparse import Namespace
from jsonargparse import Path as JSONArgparsePath
from omegaconf import DictConfig, ListConfig, OmegaConf


def to_yaml(config: Namespace | ListConfig | DictConfig) -> str:
    """Converts the config to a yaml string

    Args:
        config (Namespace | ListConfig | DictConfig): Config

    Returns:
        str: YAML string
    """
    _config = config.clone() if isinstance(config, Namespace) else config.copy()
    if isinstance(_config, Namespace):
        _config = _config.as_dict()
        _config = _convert_nested_path_to_str(_config)
    return OmegaConf.to_yaml(_config)


def _convert_nested_path_to_str(config: Any) -> Any:
    """Goes over the dictionary and converts all path values to str."""
    if isinstance(config, dict):
        for key, value in config.items():
            config[key] = _convert_nested_path_to_str(value)
    elif isinstance(config, list):
        for i, item in enumerate(config):
            config[i] = _convert_nested_path_to_str(item)
    elif isinstance(config, Path | JSONArgparsePath):
        config = str(config)
    return config


def to_tuple(input_size: int | ListConfig) -> tuple[int, int]:
    """Converts int or list to a tuple.

    Example:
        >>> to_tuple(256)
        (256, 256)
        >>> to_tuple([256, 256])
        (256, 256)

    Args:
        input_size (int | ListConfig): input_size

    Raises:
        ValueError: Unsupported value type.

    Returns:
        tuple[int, int]: Tuple of input_size
    """
    ret_val: tuple[int, int]
    if isinstance(input_size, int):
        ret_val = cast(tuple[int, int], (input_size,) * 2)
    elif isinstance(input_size, ListConfig | Sequence):
        assert len(input_size) == 2, "Expected a single integer or tuple of length 2 for width and height."
        ret_val = cast(tuple[int, int], tuple(input_size))
    else:
        raise ValueError(f"Expected either int or ListConfig, got {type(input_size)}")
    return ret_val
