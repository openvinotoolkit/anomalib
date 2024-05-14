"""Get configurable parameters."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from jsonargparse import Namespace
from jsonargparse import Path as JSONArgparsePath
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)


def _convert_nested_path_to_str(config: Any) -> Any:  # noqa: ANN401
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


def to_yaml(config: Namespace | ListConfig | DictConfig) -> str:
    """Convert the config to a yaml string.

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


def to_tuple(input_size: int | ListConfig) -> tuple[int, int]:
    """Convert int or list to a tuple.

    Args:
        input_size (int | ListConfig): input_size

    Example:
        >>> to_tuple(256)
        (256, 256)
        >>> to_tuple([256, 256])
        (256, 256)

    Raises:
        ValueError: Unsupported value type.

    Returns:
        tuple[int, int]: Tuple of input_size
    """
    ret_val: tuple[int, int]
    if isinstance(input_size, int):
        ret_val = cast(tuple[int, int], (input_size,) * 2)
    elif isinstance(input_size, ListConfig | Sequence):
        if len(input_size) != 2:
            msg = "Expected a single integer or tuple of length 2 for width and height."
            raise ValueError(msg)

        ret_val = cast(tuple[int, int], tuple(input_size))
    else:
        msg = f"Expected either int or ListConfig, got {type(input_size)}"
        raise TypeError(msg)
    return ret_val


def update_config(config: DictConfig | ListConfig | Namespace) -> DictConfig | ListConfig | Namespace:
    """Update config.

    Args:
        config: Configurable parameters.

    Returns:
        DictConfig | ListConfig | Namespace: Updated config.
    """
    _show_warnings(config)

    return _update_nncf_config(config)


def _update_nncf_config(config: DictConfig | ListConfig) -> DictConfig | ListConfig:
    """Set the NNCF input size based on the value of the crop_size parameter in the configurable parameters object.

    Args:
        config (DictConfig | ListConfig): Configurable parameters of the current run.

    Returns:
        DictConfig | ListConfig: Updated configurable parameters in DictConfig object.
    """
    if "optimization" in config and "nncf" in config.optimization:
        if "input_info" not in config.optimization.nncf:
            config.optimization.nncf["input_info"] = {"sample_size": None}
        config.optimization.nncf.input_info.sample_size = [1, 3, 10, 10]
        if config.optimization.nncf.apply and "update_config" in config.optimization.nncf:
            return OmegaConf.merge(config, config.optimization.nncf.update_config)
    return config


def _show_warnings(config: DictConfig | ListConfig | Namespace) -> None:
    """Show warnings if any based on the configuration settings.

    Args:
        config (DictConfig | ListConfig | Namespace): Configurable parameters for the current run.
    """
    if "clip_length_in_frames" in config.data and config.data.init_args.clip_length_in_frames > 1:
        logger.warning(
            "Anomalib's models and visualizer are currently not compatible with video datasets with a clip length > 1. "
            "Custom changes to these modules will be needed to prevent errors and/or unpredictable behaviour.",
        )
    if (
        "devices" in config.trainer
        and (config.trainer.devices is None or config.trainer.devices != 1)
        and config.trainer.accelerator != "cpu"
    ):
        logger.warning("Anomalib currently does not support multi-gpu training. Setting devices to 1.")
        config.trainer.devices = 1
