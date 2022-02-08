"""Utils to update configuration files."""

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

from typing import List

from omegaconf import DictConfig


def flatten_sweep_params(params_dict: DictConfig) -> DictConfig:
    """Flatten the nested parameters section of the config object.

    Args:
        params_dict: DictConfig: The dictionary containing the hpo parameters in the original, nested, structure.

    Returns:
        flattened version of the parameter dictionary.
    """

    def process_params(nested_params: DictConfig, keys: List[str], flattened_params: DictConfig):
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
                process_params(cfg, keys + [str(name)], flattened_params)
            else:
                key = ".".join(keys + [str(name)])
                flattened_params[key] = cfg

    flattened_params_dict = DictConfig({})
    process_params(params_dict, [], flattened_params_dict)

    return flattened_params_dict
