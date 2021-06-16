"""
Load Anomaly Model
"""
from typing import Type

from omegaconf import DictConfig

from .dfkde.model import DFKDEModel
from .padim.model import PADIMModel
from .stfpm.model import STFPMModel


def get_model(config: DictConfig):
    """Load model from the configuration file.

    Args:
      config: Configuration file
      config: DictConfig:

    Returns:
      Anomaly Model

    """
    model: Type[object]
    if config.model.name == "padim":
        model = PADIMModel
    elif config.model.name == "stfpm":
        model = STFPMModel
    elif config.model.name == "dfkde":
        model = DFKDEModel
    else:
        raise ValueError("Unknown model name!")

    # if config.model.name == "padim":
    #     model = PADIMModel(config)
    # elif config.model.name == "stfpm":
    #     model = STFPMModel(config)
    # elif config.model.name == "dfkde":
    #     model = DFKDEModel(config)
    # else:
    #     raise ValueError("Unknown model name!")
    return model(config)
