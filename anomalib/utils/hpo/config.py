"""Utils to update configuration files."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from omegaconf import DictConfig


def flatten_hpo_params(params_dict: DictConfig) -> DictConfig:
    """Flatten the nested hpo parameter section of the config object.

    Args:
        params_dict: DictConfig: The dictionary containing the hpo parameters in the original, nested, structure.

    Returns:
        flattened version of the parameter dictionary.
    """

    def process_params(nested_params: DictConfig, keys: list[str], flattened_params: DictConfig) -> None:
        """Flatten nested dictionary till the time it reaches the hpo params.

        Recursive helper function that traverses the nested config object and stores the leaf nodes in a flattened
        dictionary.

        Args:
            nested_params: DictConfig: config object containing the original parameters.
            keys: list[str]: list of keys leading to the current location in the config.
            flattened_params: DictConfig: Dictionary in which the flattened parameters are stored.
        """
        if len({"values", "min", "max"}.intersection(nested_params.keys())) > 0:
            key = ".".join(keys)
            flattened_params[key] = nested_params
        else:
            for name, cfg in nested_params.items():
                if isinstance(cfg, DictConfig):
                    process_params(cfg, keys + [str(name)], flattened_params)

    flattened_params_dict = DictConfig({})
    process_params(params_dict, [], flattened_params_dict)

    return flattened_params_dict
