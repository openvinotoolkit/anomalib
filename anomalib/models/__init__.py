"""
Load Anomaly Model
"""
import os
from importlib import import_module
from typing import List, Union

from omegaconf import DictConfig, ListConfig
from torch import load

from anomalib.core.model import AnomalyModule


def get_model(config: Union[DictConfig, ListConfig]) -> AnomalyModule:
    """Load model from the configuration file.
    Works only when the convention for model naming is followed.

    The convention for writing model classes is
    `anomalib.models.<model_name>.model.<Model_name>Lightning`
    `anomalib.models.stfpm.model.StfpmLightning`

    and for OpenVINO
    `anomalib.models.<model-name>.model.<Model_name>OpenVino`
    `anomalib.models.stfpm.model.StfpmOpenVino`

    Args:
        config (Union[DictConfig, ListConfig]): Config.yaml loaded using OmegaConf

    Raises:
        ValueError: If unsupported model is passed

    Returns:
        AnomalyModule: Anomaly Model
    """
    openvino_model_list: List[str] = ["stfpm"]
    torch_model_list: List[str] = ["padim", "stfpm", "dfkde", "dfm", "patchcore"]
    model: AnomalyModule

    if config.openvino:
        if config.model.name in openvino_model_list:
            module = import_module(f"anomalib.models.{config.model.name}.model")
            model = getattr(module, f"{config.model.name.capitalize()}OpenVino")
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
