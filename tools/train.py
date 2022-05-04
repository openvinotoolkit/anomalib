"""Anomalib Traning Script.

This script reads the name of the model or config file from command
line, train/test the anomaly model to get quantitative and qualitative
results.
"""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import logging
import warnings
from argparse import ArgumentParser, Namespace

from pytorch_lightning import Trainer, seed_everything

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.models import get_model
from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
from anomalib.utils.loggers import configure_logger, get_experiment_logger

logger = logging.getLogger("anomalib")


def get_args() -> Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="padim", help="Name of the algorithm to train/test")
    # --model_config_path will be deprecated in 0.2.8 and removed in 0.2.9
    parser.add_argument("--model_config_path", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--config", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--log-level", type=str, default="INFO", help="<DEBUG, INFO, WARNING, ERROR>")

    args = parser.parse_args()
    if args.model_config_path is not None:
        warnings.warn(
            message="--model_config_path will be deprecated in v0.2.8 and removed in v0.2.9. Use --config instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        args.config = args.model_config_path

    return args


def train():
    """Train an anomaly classification or segmentation model based on a provided configuration file."""
    args = get_args()
    configure_logger(level=args.log_level)

    config = get_configurable_parameters(model_name=args.model, config_path=args.config)
    if config.project.seed != 0:
        seed_everything(config.project.seed)

    if args.log_level == "ERROR":
        warnings.filterwarnings("ignore")

    logger.info("Loading the datamodule")
    datamodule = get_datamodule(config)

    logger.info("Loading the model.")
    model = get_model(config)

    logger.info("Loading the experiment logger(s)")
    experiment_logger = get_experiment_logger(config)

    logger.info("Loading the callbacks")
    callbacks = get_callbacks(config)

    trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=callbacks)
    logger.info("Training the model.")
    trainer.fit(model=model, datamodule=datamodule)

    logger.info("Loading the best model weights.")
    load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)
    trainer.callbacks.insert(0, load_model_callback)

    logger.info("Testing the model.")
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    train()
