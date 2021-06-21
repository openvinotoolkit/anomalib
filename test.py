"""
Test
This script performs inference on the test dataset and saves the output
    visualizations into a directory.
"""
from argparse import ArgumentParser, Namespace

from pytorch_lightning import Trainer

from anomalib.config.config import get_configurable_parameters
from anomalib.datasets import get_datamodule
from anomalib.models import get_model


def get_args() -> Namespace:
    """
    get_args [summary]

    Returns:
        Namespace: CLI arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="stfpm", help="Name of the algorithm to train/test")
    parser.add_argument("--model_config_path", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--weight_file", type=str, default="weights/model.ckpt")
    parser.add_argument("--openvino", type=bool, default=False)

    return parser.parse_args()


args = get_args()
config = get_configurable_parameters(
    model_name=args.model,
    model_config_path=args.model_config_path,
    weight_file=args.weight_file,
    openvino=args.openvino,
)
datamodule = get_datamodule(config)
model = get_model(config)

trainer = Trainer(callbacks=model.callbacks, **config.trainer)
trainer.test(model=model, datamodule=datamodule)
