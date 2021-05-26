from .padim.model import PADIMModel
from .stfpm import STFPMModel
from .dfkde import DFKDEModel


def get_model(config):
    if config.model.name == "padim":
        model = PADIMModel(config)
    elif config.model.name == "stfpm":
        model = STFPMModel(config)
    elif config.model.name == "dfkde":
        model = DFKDEModel(config)
    else:
        raise ValueError("Unknown model name!")

    return model
