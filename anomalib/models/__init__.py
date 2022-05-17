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
from anomalib.models.dfm import DfmLightning
from anomalib.models.ganomaly import GanomalyLightning


def get_model(config: Union[DictConfig, ListConfig]) -> AnomalyModule:
    """Load model from the configuration file.

    Works only when the convention for model naming is followed.

    The convention for writing model classes is
    `anomalib.models.<model_name>.model.<Model_name>Lightning`
    `anomalib.models.stfpm.model.StfpmLightning`

    Args:
        config (Union[DictConfig, ListConfig]): Config.yaml loaded using OmegaConf

    Raises:
        ValueError: If unsupported model is passed

    Returns:
        AnomalyModule: Anomaly Model
    """
    model_list: List[str] = ["cflow", "dfkde", "padim", "patchcore", "stfpm"]
    model: AnomalyModule

    if config.model.name in model_list:
        module = import_module(f"anomalib.models.{config.model.name}")
        model = getattr(module, f"{config.model.name.capitalize()}Lightning")(config)

    elif config.model.name == "dfm":
        model = DfmLightning(
            adaptive_threshold=config.model.threshold.adaptive,
            default_image_threshold=config.model.threshold.image_default,
            backbone=config.model.backbone,
            layer=config.model.layer,
            pooling_kernel_size=config.model.pooling_kernel_size,
            pca_level=config.model.pca_level,
            score_type=config.model.score_type,
        )

    elif config.model.name == "ganomaly":
        model = GanomalyLightning(
            adaptive_threshold=config.model.threshold.adaptive,
            default_image_threshold=config.model.threshold.image_default,
            batch_size=config.dataset.train_batch_size,
            input_size=config.model.input_size,
            n_features=config.model.n_features,
            latent_vec_size=config.model.latent_vec_size,
            extra_layers=config.model.extra_layers,
            add_final_conv_layer=config.model.add_final_conv,
            wadv=config.model.wadv,
            wcon=config.model.wcon,
            wenc=config.model.wenc,
            learning_rate=config.model.lr,
            beta1=config.model.beta1,
            beta2=config.model.beta2,
            early_stopping_metric=config.model.early_stopping.metric,
            early_stopping_patience=config.model.early_stopping.patience,
            early_stopping_mode=config.model.early_stopping.mode,
        )

    else:
        raise ValueError(f"Unknown model {config.model.name}!")

    if "init_weights" in config.keys() and config.init_weights:
        model.load_state_dict(load(os.path.join(config.project.path, config.init_weights))["state_dict"], strict=False)

    return model
