from .stfpm import STFPMModel, STFPMLightning
from .dfkde import DFKDEModel

def get_model(config, *args):
    if config.model == "stfpm":
        model = STFPMLightning(config, *args)
    elif config.model == "dfkde":
        model = DFKDEModel(config)
    else:
        raise ValueError("Unknown model name!")

    return model
