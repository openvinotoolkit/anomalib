from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything

from anomalib.config.config import get_configurable_parameters
from anomalib.datasets import get_datamodule
from anomalib.models import get_model


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="padim", help="Name of the algorithm to train/test")
    parser.add_argument("--model_config_path", type=str, required=False, help="Path to a model config file")

    return parser.parse_args()

args = get_args()
config = get_configurable_parameters(model_name=args.model, model_config_path=args.model_config_path)

if config.project.seed != 0:
    seed_everything(config.project.seed)

datamodule = get_datamodule(config)
model = get_model(config)

trainer = Trainer(callbacks=model.callbacks, **config.trainer)

trainer.fit(model=model, datamodule=datamodule)
trainer.test(model=model, datamodule=datamodule)
