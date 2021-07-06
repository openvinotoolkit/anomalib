"""
Load Anomaly Model
"""
from typing import Type, Union

from omegaconf import DictConfig, ListConfig

from .dfkde.model import DFKDELightning
from .padim.model import PADIMLightning
from .stfpm.model import STFPMLightning, STFPMOpenVino


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
        else:
            raise ValueError("Unknown model name!")

    return model(config)
