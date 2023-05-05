"""Test This script performs inference on the test dataset and saves the output visualizations into a directory."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser, Namespace

from pytorch_lightning import Trainer, seed_everything

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks


def get_args() -> Namespace:
    """Get CLI arguments.

    Returns:
        Namespace: CLI arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="stfpm", help="Name of the algorithm to train/test")
    parser.add_argument("--config", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--weight_file", type=str, default="weights/model.ckpt")

    args = parser.parse_args()
    return args


def test():
    """Test an anomaly classification and segmentation model that is initially trained via `tools/train.py`.

    The script is able to write the results into both filesystem and a logger such as Tensorboard.
    """
    args = get_args()
    config = get_configurable_parameters(
        model_name=args.model,
        config_path=args.config,
        weight_file=args.weight_file,
    )

    if config.project.seed:
        seed_everything(config.project.seed)

    datamodule = get_datamodule(config)
    model = get_model(config)

    callbacks = get_callbacks(config)

    trainer = Trainer(callbacks=callbacks, **config.trainer)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    test()
