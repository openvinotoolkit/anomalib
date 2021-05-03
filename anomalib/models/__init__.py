from .stfpm import STFPMModel


def get_model(args):
    if args.model == "stfpm":
        model = STFPMModel(args)
    else:
        raise ValueError("Unknown model name!")

    return model
