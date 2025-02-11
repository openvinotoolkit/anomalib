"""Configuration utilities.

This module contains utility functions for handling configuration objects, including:
- Converting between different configuration formats (dict, Namespace, DictConfig)
- Flattening and nesting dictionaries
- Converting paths and values
- Updating configurations
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Iterable, Sequence, ValuesView
from pathlib import Path
from typing import Any, cast

from jsonargparse import Namespace
from jsonargparse import Path as JSONArgparsePath
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)


def _convert_nested_path_to_str(config: Any) -> Any:  # noqa: ANN401
    """Convert all path values to strings recursively in a configuration object.

    This function traverses a configuration object and converts any ``Path`` or
    ``JSONArgparsePath`` objects to string representations. It handles nested
    dictionaries and lists recursively.

    Args:
        config: Configuration object that may contain path values. Can be a
            dictionary, list, Path object, or other types.

    Returns:
        Any: Configuration with all path values converted to strings. The returned
            object maintains the same structure as the input, with only path
            values converted to strings.

    Examples:
        >>> from pathlib import Path
        >>> config = {
        ...     "model_path": Path("/path/to/model"),
        ...     "data": {
        ...         "train_path": Path("/data/train"),
        ...         "val_path": Path("/data/val")
        ...     }
        ... }
        >>> converted = _convert_nested_path_to_str(config)
        >>> print(converted["model_path"])
        /path/to/model
        >>> print(converted["data"]["train_path"])
        /data/train
    """
    if isinstance(config, dict):
        for key, value in config.items():
            config[key] = _convert_nested_path_to_str(value)
    elif isinstance(config, list):
        for i, item in enumerate(config):
            config[i] = _convert_nested_path_to_str(item)
    elif isinstance(config, Path | JSONArgparsePath):
        config = str(config)
    return config


def to_nested_dict(config: dict) -> dict:
    """Convert a flattened dictionary to a nested dictionary.

    This function takes a dictionary with dot-separated keys and converts it into a nested
    dictionary structure. Keys containing dots (`.`) are split and used to create nested
    dictionaries.

    Args:
        config: Flattened dictionary where keys can contain dots to indicate nesting
               levels. For example, ``"dataset.category"`` will become
               ``{"dataset": {"category": ...}}``.

    Returns:
        dict: A nested dictionary where dot-separated keys in the input are converted
              to nested dictionary structures. Keys without dots remain at the top
              level.

    Examples:
        >>> config = {
        ...     "dataset.category": "bottle",
        ...     "dataset.image_size": 224,
        ...     "model_name": "padim"
        ... }
        >>> result = to_nested_dict(config)
        >>> print(result["dataset"]["category"])
        bottle
        >>> print(result["dataset"]["image_size"])
        224
        >>> print(result["model_name"])
        padim

    Note:
        - The function preserves the original values while only restructuring the keys
        - Non-dot keys are kept as-is at the root level
        - Empty key segments (e.g. ``"dataset..name"``) are handled as literal keys
    """
    out: dict[str, Any] = {}
    for key, value in config.items():
        keys = key.split(".")
        _dict = out
        for k in keys[:-1]:
            _dict = _dict.setdefault(k, {})
        _dict[keys[-1]] = value
    return out


def to_yaml(config: Namespace | ListConfig | DictConfig) -> str:
    """Convert configuration object to YAML string.

    This function takes a configuration object and converts it to a YAML formatted string.
    It handles different configuration object types including ``Namespace``,
    ``ListConfig``, and ``DictConfig``.

    Args:
        config: Configuration object to convert. Can be one of:
            - ``Namespace``: A namespace object from OmegaConf
            - ``ListConfig``: A list configuration from OmegaConf
            - ``DictConfig``: A dictionary configuration from OmegaConf

    Returns:
        str: Configuration as YAML formatted string

    Examples:
        >>> from omegaconf import DictConfig
        >>> config = DictConfig({"model": "padim", "dataset": {"name": "mvtecad"}})
        >>> yaml_str = to_yaml(config)
        >>> print(yaml_str)
        model: padim
        dataset:
          name: mvtec

    Note:
        - For ``Namespace`` objects, the function first converts to dictionary format
        - Nested paths in the configuration are converted to strings
        - The original configuration object is not modified
    """
    _config = config.clone() if isinstance(config, Namespace) else config.copy()
    if isinstance(_config, Namespace):
        _config = _config.as_dict()
        _config = _convert_nested_path_to_str(_config)
    return OmegaConf.to_yaml(_config)


def to_tuple(input_size: int | ListConfig) -> tuple[int, int]:
    """Convert input size to a tuple of (height, width).

    This function takes either a single integer or a sequence of two integers and
    converts it to a tuple representing image dimensions (height, width). If a single
    integer is provided, it is used for both dimensions.

    Args:
        input_size: Input size specification. Can be either:
            - A single ``int`` that will be used for both height and width
            - A ``ListConfig`` or sequence containing exactly 2 integers for height
              and width

    Returns:
        tuple[int, int]: A tuple of ``(height, width)`` dimensions

    Examples:
        Create a square tuple from single integer:

        >>> to_tuple(256)
        (256, 256)

        Create a tuple from list of dimensions:

        >>> to_tuple([256, 256])
        (256, 256)

    Raises:
        ValueError: If ``input_size`` is a sequence without exactly 2 elements
        TypeError: If ``input_size`` is neither an integer nor a sequence of
            integers

    Note:
        When using a sequence input, the first value is interpreted as height and
        the second as width.
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


def convert_valuesview_to_tuple(values: ValuesView) -> list[tuple]:
    """Convert ``ValuesView`` to list of tuples for parameter combinations.

    This function takes a ``ValuesView`` object and converts it to a list of tuples
    that can be used for creating parameter combinations. It is particularly useful
    when working with ``itertools.product`` to generate all possible parameter
    combinations.

    The function handles both iterable and non-iterable values:
    - Iterable values (except strings) are converted to tuples
    - Non-iterable values and strings are wrapped in single-element tuples

    Args:
        values: A ``ValuesView`` object containing parameter values to convert

    Returns:
        list[tuple]: A list of tuples where each tuple contains parameter values.
            Single values are wrapped in 1-element tuples.

    Examples:
        Create parameter combinations from a config:

        >>> params = DictConfig({
        ...     "dataset.category": [
        ...         "bottle",
        ...         "cable",
        ...     ],
        ...     "dataset.image_size": 224,
        ...     "model_name": ["padim"],
        ... })
        >>> convert_valuesview_to_tuple(params.values())
        [('bottle', 'cable'), (224,), ('padim',)]

        Use with ``itertools.product`` to get all combinations:

        >>> list(itertools.product(*convert_valuesview_to_tuple(params.values())))
        [('bottle', 224, 'padim'), ('cable', 224, 'padim')]

    Note:
        Strings are treated as non-iterable values even though they are technically
        iterable in Python. This prevents unwanted character-by-character splitting.
    """
    return_list = []
    for value in values:
        if isinstance(value, Iterable) and not isinstance(value, str):
            return_list.append(tuple(value))
        else:
            return_list.append((value,))
    return return_list


def flatten_dict(config: dict, prefix: str = "") -> dict:
    """Flatten a nested dictionary using dot notation.

    Takes a nested dictionary and flattens it into a single-level dictionary where
    nested keys are joined using dot notation. This is useful for converting
    hierarchical configurations into a flat format.

    Args:
        config: Nested dictionary to flatten. Can contain arbitrary levels of
            nesting.
        prefix: Optional string prefix to prepend to all flattened keys. Defaults
            to empty string.

    Returns:
        dict: Flattened dictionary where nested keys are joined with dots.
            For example, ``{"a": {"b": 1}}`` becomes ``{"a.b": 1}``.

    Examples:
        Basic nested dictionary flattening:

        >>> config = {
        ...     "dataset": {
        ...         "category": "bottle",
        ...         "image_size": 224
        ...     },
        ...     "model_name": "padim"
        ... }
        >>> flattened = flatten_dict(config)
        >>> print(flattened)  # doctest: +SKIP
        {
            'dataset.category': 'bottle',
            'dataset.image_size': 224,
            'model_name': 'padim'
        }

        With custom prefix:

        >>> flattened = flatten_dict(config, prefix="config.")
        >>> print(flattened)  # doctest: +SKIP
        {
            'config.dataset.category': 'bottle',
            'config.dataset.image_size': 224,
            'config.model_name': 'padim'
        }
    """
    out = {}
    for key, value in config.items():
        if isinstance(value, dict):
            out.update(flatten_dict(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def namespace_from_dict(container: dict) -> Namespace:
    """Convert a dictionary to a Namespace object recursively.

    This function takes a dictionary and recursively converts it and all nested
    dictionaries into ``Namespace`` objects. This is useful for accessing dictionary
    keys as attributes.

    Args:
        container: Dictionary to convert into a ``Namespace`` object. Can contain
            arbitrary levels of nesting.

    Returns:
        ``Namespace`` object with equivalent structure to input dictionary. Nested
        dictionaries are converted to nested ``Namespace`` objects.

    Examples:
        Basic dictionary conversion:

        >>> container = {
        ...     "dataset": {
        ...         "category": "bottle",
        ...         "image_size": 224,
        ...     },
        ...     "model_name": "padim",
        ... }
        >>> namespace = namespace_from_dict(container)
        >>> namespace.dataset.category
        'bottle'
        >>> namespace.model_name
        'padim'

        The returned object allows attribute-style access:

        >>> namespace.dataset.image_size
        224

    Note:
        All dictionary keys must be valid Python identifiers to be accessed as
        attributes in the resulting ``Namespace`` object.
    """
    output = Namespace()
    for k, v in container.items():
        if isinstance(v, dict):
            setattr(output, k, namespace_from_dict(v))
        else:
            setattr(output, k, v)
    return output


def dict_from_namespace(container: Namespace) -> dict:
    """Convert a Namespace object to a dictionary recursively.

    This function takes a ``Namespace`` object and recursively converts it and all nested
    ``Namespace`` objects into dictionaries. This is useful for serializing ``Namespace``
    objects or converting them to a format that can be easily saved or transmitted.

    Args:
        container: ``Namespace`` object to convert into a dictionary. Can contain
            arbitrary levels of nesting.

    Returns:
        Dictionary with equivalent structure to input ``Namespace``. Nested
        ``Namespace`` objects are converted to nested dictionaries.

    Examples:
        Basic namespace conversion:

        >>> from jsonargparse import Namespace
        >>> ns = Namespace()
        >>> ns.a = 1
        >>> ns.b = Namespace()
        >>> ns.b.c = 2
        >>> dict_from_namespace(ns)
        {'a': 1, 'b': {'c': 2}}

        The function handles arbitrary nesting:

        >>> ns = Namespace()
        >>> ns.x = Namespace()
        >>> ns.x.y = Namespace()
        >>> ns.x.y.z = 3
        >>> dict_from_namespace(ns)
        {'x': {'y': {'z': 3}}}

    Note:
        This function is the inverse of :func:`namespace_from_dict`. Together they
        provide bidirectional conversion between dictionaries and ``Namespace``
        objects.
    """
    output = {}
    for k, v in container.__dict__.items():
        if isinstance(v, Namespace):
            output[k] = dict_from_namespace(v)
        else:
            output[k] = v
    return output


def update_config(config: DictConfig | ListConfig | Namespace) -> DictConfig | ListConfig | Namespace:
    """Update configuration with warnings and NNCF settings.

    This function processes the provided configuration by:
        - Showing relevant configuration-specific warnings via ``_show_warnings``
        - Updating NNCF (Neural Network Compression Framework) settings via
          ``_update_nncf_config``

    Args:
        config: Configuration object to update. Can be either a ``DictConfig``,
            ``ListConfig``, or ``Namespace`` instance containing model and training
            parameters.

    Returns:
        Updated configuration with any NNCF-specific modifications applied. Returns
        the same type as the input configuration.

    Examples:
        >>> from omegaconf import DictConfig
        >>> config = DictConfig({"optimization": {"nncf": {"apply": True}}})
        >>> updated = update_config(config)

        >>> from jsonargparse import Namespace
        >>> config = Namespace(data={"clip_length_in_frames": 1})
        >>> updated = update_config(config)

    Note:
        This function is typically called after loading the initial configuration
        but before using it for model training or inference.
    """
    _show_warnings(config)

    return _update_nncf_config(config)


def _update_nncf_config(config: DictConfig | ListConfig) -> DictConfig | ListConfig:
    """Update NNCF configuration with input size settings.

    This function updates the Neural Network Compression Framework (NNCF)
    configuration by setting default input size parameters if they are not already
    specified. It also handles merging any NNCF-specific configuration updates.

    The function checks if NNCF optimization settings exist in the config and adds
    default input shape information of ``[1, 3, 10, 10]`` if not present. If NNCF
    is enabled and contains update configuration, it merges those updates.

    Args:
        config: Configuration object containing NNCF settings. Must be either a
            ``DictConfig`` or ``ListConfig`` instance.

    Returns:
        ``DictConfig`` or ``ListConfig`` with updated NNCF configuration settings.

    Example:
        >>> from omegaconf import DictConfig
        >>> config = DictConfig({
        ...     "optimization": {
        ...         "nncf": {
        ...             "apply": True,
        ...             "input_info": {"sample_size": [1, 3, 224, 224]}
        ...         }
        ...     }
        ... })
        >>> updated = _update_nncf_config(config)

    Note:
        The default input size of ``[1, 3, 10, 10]`` represents:
        - Batch size of 1
        - 3 input channels (RGB)
        - Height and width of 10 pixels
    """
    if "optimization" in config and "nncf" in config.optimization:
        if "input_info" not in config.optimization.nncf:
            config.optimization.nncf["input_info"] = {"sample_size": None}
        config.optimization.nncf.input_info.sample_size = [1, 3, 10, 10]
        if config.optimization.nncf.apply and "update_config" in config.optimization.nncf:
            return OmegaConf.merge(config, config.optimization.nncf.update_config)
    return config


def _show_warnings(config: DictConfig | ListConfig | Namespace) -> None:
    """Show configuration-specific warnings.

    This function checks the provided configuration for conditions that may cause
    issues and displays appropriate warning messages. Currently checks for:

        - Video clip length compatibility issues with models and visualizers

    Args:
        config: Configuration object to check for warning conditions. Can be one of:
            - ``DictConfig``
            - ``ListConfig``
            - ``Namespace``

    Example:
        >>> from omegaconf import DictConfig
        >>> config = DictConfig({
        ...     "data": {
        ...         "init_args": {"clip_length_in_frames": 2}
        ...     }
        ... })
        >>> _show_warnings(config)  # Will show video clip length warning

    Note:
        The function currently focuses on video-related configuration warnings,
        specifically checking the ``clip_length_in_frames`` parameter in the data
        configuration section.
    """
    if "clip_length_in_frames" in config.data and config.data.init_args.clip_length_in_frames > 1:
        logger.warning(
            "Anomalib's models and visualizer are currently not compatible with video datasets with a clip length > 1. "
            "Custom changes to these modules will be needed to prevent errors and/or unpredictable behaviour.",
        )
