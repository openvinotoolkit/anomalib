"""
Load Anomaly Model
"""
from omegaconf import DictConfig

from .dfkde.model import DFKDELightning
from .padim.model import PADIMLightning
from .stfpm.model import STFPMLightning
from .stfpm.model_openvino import STFPMOpenVino


def get_model(config: DictConfig):
    """
    Load model from the configuration file.

    :param config: Configuration file
    :return: Anomaly Model
    """
    if config.openvino:
        if config.model.name == "stfpm":
            model = STFPMOpenVino(config)
        else:
            raise ValueError("Unknown model name for OpenVINO model!")
    else:
        if config.model.name == "padim":
            model = PADIMLightning(config)
        elif config.model.name == "stfpm":
            model = STFPMLightning(config)
        elif config.model.name == "dfkde":
            model = DFKDELightning(config)
        else:
            raise ValueError("Unknown model name!")

    return model
