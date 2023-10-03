"""Utilities to help serialize/deserialize the config."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path, PosixPath
from typing import Type, cast

import yaml
from jsonargparse import Namespace
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf._utils import OmegaConfDumper


class _YAMLDumper(OmegaConfDumper):
    path_representation_added = False

    @staticmethod
    def path_representer(dumper: yaml.Dumper, data: Path | PosixPath) -> yaml.ScalarNode:
        return dumper.represent_scalar(
            yaml.resolver.BaseResolver.DEFAULT_SCALAR_TAG,
            str(data),
        )


def _get_dumper() -> Type[_YAMLDumper]:
    """YAML serializer for path."""
    if not _YAMLDumper.str_representer_added:
        _YAMLDumper.add_representer(str, _YAMLDumper.str_representer)
        _YAMLDumper.str_representer_added = True
    if not _YAMLDumper.path_representation_added:
        _YAMLDumper.add_representer(Path, _YAMLDumper.path_representer)
        _YAMLDumper.add_representer(PosixPath, _YAMLDumper.path_representer)
        _YAMLDumper.path_representation_added = True
    return _YAMLDumper


def to_yaml(config: Namespace | ListConfig | DictConfig) -> str:
    """Converts the config to a yaml string

    Args:
        config (Namespace | ListConfig | DictConfig): Config

    Returns:
        str: YAML string
    """
    _config = config.clone() if isinstance(config, Namespace) else config.copy()
    if "config" in _config.keys():
        del _config["config"]

    if isinstance(_config, Namespace):
        _config = OmegaConf.create(_config.as_dict())
    container = OmegaConf.to_container(_config, enum_to_str=True)
    return yaml.dump(container, default_flow_style=False, allow_unicode=True, sort_keys=False, Dumper=_get_dumper())


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
    elif isinstance(input_size, ListConfig):
        assert len(input_size) == 2, "Expected a single integer or tuple of length 2 for width and height."
        ret_val = cast(tuple[int, int], tuple(input_size))
    else:
        raise ValueError(f"Expected either int or ListConfig, got {type(input_size)}")
    return ret_val
