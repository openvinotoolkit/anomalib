"""
Load Anomaly Model
"""
from typing import Type, Union
import os

from torch import load
from omegaconf import DictConfig, ListConfig

from .dfkde.model import DFKDELightning
from .padim.model import PADIMLightning
from .patchcore.model import PatchcoreLightning
from .stfpm.model import STFPMLightning, STFPMOpenVino
from .dfm.model import DFMLightning


def get_model(config: Union[DictConfig, ListConfig]):
    """Load model from the configuration file.

    Args:
      config: Configuration file
      config: DictConfig:

    Returns:
      Anomaly Model

    """
    model: Type[object]

    if config.openvino:
        if config.model.name == "stfpm":
            model = STFPMOpenVino
        else:
            raise ValueError("Unknown model name for OpenVINO model!")
    else:
        if config.model.name == "padim":
            model = PADIMLightning
        elif config.model.name == "stfpm":
            model = STFPMLightning
        elif config.model.name == "dfkde":
            model = DFKDELightning
        elif config.model.name == "dfm":
            model = DFMLightning
        elif config.model.name == "patchcore":
            model = PatchcoreLightning
        else:
            raise ValueError("Unknown model name!")

    model = model(config)

    if "init_weights" in config.keys():
        model.load_state_dict(load(os.path.join(config.project.path, config.init_weights))['state_dict'], strict=False)

    return model
