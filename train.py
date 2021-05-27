from argparse import ArgumentParser

from pytorch_lightning import Trainer
import torch
import nncf
from nncf import NNCFConfig, create_compressed_model, load_state, register_default_init_args
import json

from anomalib.config import get_configurable_parameters
from anomalib.datasets import get_datamodule
from anomalib.models import get_model


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="stfpm", help="Name of the algorithm to train/test")
    parser.add_argument("--model_config_path", type=str, required=False, help="Path to a model config file")
    args = parser.parse_args()
    return args


with open("anomalib/models/stfpm/nncf.json") as f:
    config_dict = json.load(f)

args = get_args()

config = get_configurable_parameters(model_name=args.model, model_config_path=args.model_config_path)
datamodule = get_datamodule(config.dataset)
datamodule.setup()
train_loader = datamodule.train_dataloader()
model = get_model(config.model, train_loader)

trainer = Trainer(callbacks=model.callbacks, **config.trainer)

trainer.fit(model=model, datamodule=datamodule)
