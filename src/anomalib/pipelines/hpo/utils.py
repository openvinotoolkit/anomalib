"""HPO utils."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from functools import reduce
from typing import Any


def flatten_hpo_params(params_dict: dict) -> dict:
    """Flatten the nested sections except hpo specific sections.

    Args:
        params_dict (dict): Nested dictionary.

    Example:
        >>> params_dict = {
                "data": {
                    "class_path": "MVTec",
                    "init_args": {
                        "category": "bottle",
                        "image_size": {
                            "type": "discrete",
                            "values": [128, 256],
                        }
                    }
                },
                "model": {
                    "class_path": "Padim",
                    "init_args": {
                        "backbone": {
                            "type": "categorical",
                            "values": ["resnet18", "wide_resnet50_2"],
                        }
                    }
                }
            }
        >>> flatten_hpo_params(params_dict)
        {
            "data.class_path": MVTec,
            "data.init_args.category": "bottle",
            "data.init_args.image_size": {
                "type": "discrete",
                "values": [128, 256],
            },
            "model.class_path": "Padim",
            "model.init_args.backbone": {
                "type": "categorical",
                "values": ["resnet18", "wide_resnet50_2"],
        }

    Returns:
        dict: Flattened dictionary.
    """

    def _process_params(nested_params: dict, keys: list[str], flat_params: dict) -> None:
        if len({"values", "min", "max"}.intersection(nested_params.keys())) > 0:
            flat_params[".".join(keys)] = nested_params
        else:
            for key, value in nested_params.items():
                if isinstance(value, dict):
                    _process_params(value, [*keys, str(key)], flat_params)

    flattened_params: dict[str, Any] = {}
    _process_params(params_dict, [], flattened_params)
    return flattened_params


def set_in_nested_dict(nested_dict: dict, flat_dict: dict) -> dict:
    """Set the values in a copy of the nested dictionary.

    Args:
        nested_dict (dict): Nested dictionary whose values are overridden.
        flat_dict (dict): Flattened dictionary.

    Example:
        >>> nested_dict = {
                "data": {
                    "class_path": "MVTec",
                    "init_args": {
                        "category": "bottle",
                        "image_size": {
                            "type": "discrete",
                            "values": [128, 256],
                        }
                    }
                },
            }
        >>> flat_dict = {
                "data.init_args.category": "carpet",
                "data.init_args.image_size": 128,
                }
        >>> set_in_nested_dict(nested_dict, flat_dict)
        {
            "data": {
                "class_path": "MVTec",
                "init_args": {
                    "category": "carpet",
                    "image_size": 128,
                },
            },
        }
    """
    nested_dict = deepcopy(nested_dict)
    for key, value in flat_dict.items():
        keys = key.split(".")
        reduce(lambda d, k: d.get(k, None), keys[:-1], nested_dict)[keys[-1]] = value
    return nested_dict
