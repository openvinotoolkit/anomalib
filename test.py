from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer

from anomalib.config import get_configurable_parameters
from anomalib.datasets import get_datamodule
from anomalib.models import get_model


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="stfpm", help="Name of the algorithm to train/test")
    parser.add_argument("--model_config_path", type=str, required=False, help="Path to a model config file")
    args = parser.parse_args()
    return args


args = get_args()
config = get_configurable_parameters(model_name=args.model)

datamodule = get_datamodule(config.dataset)

# TODO: load_from_checkpoint doesn't properly load the weights!!!
# model.load_from_checkpoint(checkpoint_path="./results/weights/new_model.ckpt")
model = get_model(config.model)
model.load_state_dict(torch.load("./results/weights/model.ckpt")["state_dict"])

trainer = Trainer(**config.trainer)
trainer.test(model, datamodule=datamodule)
