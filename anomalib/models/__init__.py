"""Load Anomaly Model."""

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

import os
from importlib import import_module
from typing import List, Union

from omegaconf import DictConfig, ListConfig
from torch import load

from anomalib.models.components import AnomalyModule

# TODO(AlexanderDokuchaev): Workaround of wrapping by NNCF.
#                           Can't not wrap `spatial_softmax2d` if use import_module.
from anomalib.models.padim.model import PadimLightning  # noqa: F401


def get_model(config: Union[DictConfig, ListConfig]) -> AnomalyModule:
    """Load model from the configuration file.

    Works only when the convention for model naming is followed.

    The convention for writing model classes is
    `anomalib.models.<model_name>.model.<Model_name>Lightning`
    `anomalib.models.stfpm.model.StfpmLightning`

    and for OpenVINO
    `anomalib.models.<model-name>.model.<Model_name>OpenVINO`
    `anomalib.models.stfpm.model.StfpmOpenVINO`

    Args:
        config (Union[DictConfig, ListConfig]): Config.yaml loaded using OmegaConf

    Raises:
        ValueError: If unsupported model is passed

    Returns:
        AnomalyModule: Anomaly Model
    """
    openvino_model_list: List[str] = ["stfpm"]
    torch_model_list: List[str] = ["padim", "stfpm", "dfkde", "dfm", "patchcore", "cflow", "ganomaly"]
    model: AnomalyModule

    if config.openvino:
        if config.model.name in openvino_model_list:
            module = import_module(f"anomalib.models.{config.model.name}.model")
            model = getattr(module, f"{config.model.name.capitalize()}OpenVINO")
        else:
            raise ValueError(f"Unknown model {config.model.name} for OpenVINO model!")
    else:
        if config.model.name in torch_model_list:
            module = import_module(f"anomalib.models.{config.model.name}.model")
            model = getattr(module, f"{config.model.name.capitalize()}Lightning")
        else:
            raise ValueError(f"Unknown model {config.model.name}!")

    model = model(config)

    if "init_weights" in config.keys() and config.init_weights:
        model.load_state_dict(load(os.path.join(config.project.path, config.init_weights))["state_dict"], strict=False)

    return model
