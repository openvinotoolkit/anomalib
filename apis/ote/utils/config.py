"""
Configurable parameter conversion between OTE and Anomalib.
"""

# Copyright (C) 2021 Intel Corporation
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

from typing import Union

from omegaconf import DictConfig, ListConfig
from ote_sdk.configuration.configurable_parameters import ConfigurableParameters

from anomalib.config.config import get_configurable_parameters


def get_anomalib_config(ote_config: ConfigurableParameters) -> Union[DictConfig, ListConfig]:
    """
    Create an anomalib config object that matches the values specified in the OTE config.

    Args:
        ote_config: ConfigurableParameters: OTE config object parsed from configuration.yaml file
    Returns:
        Anomalib config object for the specified model type with overwritten default values.
    """
    anomalib_config = get_configurable_parameters(getattr(ote_config, "model").name.value)
    update_anomalib_config(anomalib_config, ote_config)
    return anomalib_config


def update_anomalib_config(anomalib_config: Union[DictConfig, ListConfig], ote_config: ConfigurableParameters):
    """
    Overwrite the default parameter values in the anomalib config with the values specified in the OTE config. The
    function is recursively called for each parameter group present in the OTE config.

    Args:
        anomalib_config: DictConfig: Anomalib config object
        ote_config: ConfigurableParameters: OTE config object parsed from configuration.yaml file
    """
    for param in ote_config.parameters:
        assert param in anomalib_config.keys(), f"Parameter {param} not present in anomalib config."
        sc_value = getattr(ote_config, param)
        sc_value = sc_value.value if hasattr(sc_value, "value") else sc_value
        anomalib_config[param] = sc_value
    for group in ote_config.groups:
        # Since pot_parameters are specific to OTE
        if group != "pot_parameters":
            update_anomalib_config(anomalib_config[group], getattr(ote_config, group))
