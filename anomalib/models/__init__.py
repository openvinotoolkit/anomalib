from .stfpm import STFPMModel
from .anocls import AnoCLSModel

def get_model(args):
    if args.model == "stfpm":
        model = STFPMModel(args)
    elif args.model == "anocls":
        model = AnoCLSModel(args)
    else:
        raise ValueError("Unknown model name!")

    return model
