"""
Load Anomaly Model
"""
from omegaconf import DictConfig

from .dfkde.model import DFKDEModel
from .padim.model import PADIMModel
from .stfpm.model import STFPMModel


def get_model(config: DictConfig):
    """
    Load model from the configuration file.

    :param config: Configuration file
    :return: Anomaly Model
    """
    if config.model.name == "padim":
        model = PADIMModel(config)
    elif config.model.name == "stfpm":
        model = STFPMModel(config)
    elif config.model.name == "dfkde":
        model = DFKDEModel(config)
    else:
        raise ValueError("Unknown model name!")

    return model
