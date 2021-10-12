"""Entry point for hyperparameter optimization"""
from argparse import ArgumentParser

from pytorch_lightning import seed_everything

from anomalib.config.config import get_configurable_parameters
from anomalib.hpo.sweep import run_sweep


def get_args():
    """Gets parameters from commandline"""
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="stfpm", help="Name of the algorithm to train/test")
    parser.add_argument("--model_config_path", type=str, required=False, help="Path to a model config file")
    parser.add_argument(
        "--hpo_type", type=str, default="sweep", required=False, help="Type of hyperparameter optimization."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = get_configurable_parameters(model_name=args.model, model_config_path=args.model_config_path)

    if config.project.seed != 0:
        seed_everything(config.project.seed)

    if args.hpo_type == "sweep":
        run_sweep(config=config)
    else:
        raise ValueError(f"Unknown type: {args.hpo_type} found. Available options are [sweep]")
