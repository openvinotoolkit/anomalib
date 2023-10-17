"""Anomalib Training Script.

This script reads the name of the model or config file from command
line, train/test the anomaly model to get quantitative and qualitative
results.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import warnings
from argparse import ArgumentParser, Namespace

from lightning.pytorch import seed_everything

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.data.utils import TestSplitMode
from anomalib.engine import Engine
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks
from anomalib.utils.loggers import configure_logger, get_experiment_logger

logger = logging.getLogger("anomalib")


def get_parser() -> ArgumentParser:
    """Get parser.

    Returns:
        ArgumentParser: The parser object.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="padim", help="Name of the algorithm to train/test")
    parser.add_argument("--config", type=str, required=False, help="Path to a model config file")
    parser.add_argument("--log-level", type=str, default="INFO", help="<DEBUG, INFO, WARNING, ERROR>")

    return parser


def train(args: Namespace) -> None:
    """Train an anomaly model.

    Args:
    ----
        args (Namespace): The arguments from the command line.
    """
    configure_logger(level=args.log_level)

    if args.log_level == "ERROR":
        warnings.filterwarnings("ignore")

    config = get_configurable_parameters(model_name=args.model, config_path=args.config)
    if config.get("seed_everything", None) is not None:
        seed_everything(config.seed_everything)

    datamodule = get_datamodule(config)
    model = get_model(config)
    experiment_logger = get_experiment_logger(config)
    callbacks = get_callbacks(config)

    engine = Engine(
        **config.trainer,
        logger=experiment_logger,
        callbacks=callbacks,
        normalization=config.normalization.normalization_method,
        threshold=config.metrics.threshold,
        task=config.task,
        image_metrics=config.metrics.get("image", None),
        pixel_metrics=config.metrics.get("pixel", None),
        visualization=config.visualization,
    )

    logger.info("Training the model.")
    engine.fit(model=model, datamodule=datamodule)

    if config.data.init_args.test_split_mode == TestSplitMode.NONE:
        logger.info("No test set provided. Skipping test stage.")
    else:
        logger.info("Testing the model with best model weights.")
        engine.test(model=model, datamodule=datamodule, ckpt_path=engine.trainer.checkpoint_callback.best_model_path)


if __name__ == "__main__":
    args = get_parser().parse_args()
    train(args)
