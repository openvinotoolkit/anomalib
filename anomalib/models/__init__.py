from .padim.model import PADIMModel
from .stfpm import STFPMModel
from .dfkde import DFKDEModel


def get_model(args):
    if args.model == "padim":
        model = PADIMModel(args)
    elif args.model == "stfpm":
        model = STFPMModel(args)
    elif args.model == "dfkde":
        model = DFKDEModel(args)
    else:
        raise ValueError("Unknown model name!")

    return model
