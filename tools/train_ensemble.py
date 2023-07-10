"""Anomalib Training Script for ensemble of models.

This script reads the name of the model or config file from command
line, train/test the anomaly model to get quantitative and qualitative
results.

This code is currently not very clean as it's in prototyping stage.
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import warnings
from argparse import ArgumentParser, Namespace
from itertools import product

from pytorch_lightning import Trainer, seed_everything

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.data.utils import TestSplitMode
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks
from anomalib.utils.loggers import configure_logger, get_experiment_logger

from anomalib.models.ensemble.ensemble_tiler import EnsembleTiler
from anomalib.models.ensemble.ensemble_functions import (
    TileCollater,
    update_ensemble_input_size_config,
    BasicPredictionJoiner,
    visualize_results,
)


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


def train(args: Namespace):
    """Train an anomaly model.

    Args:
        args (Namespace): The arguments from the command line.
    """

    configure_logger(level=args.log_level)

    if args.log_level == "ERROR":
        warnings.filterwarnings("ignore")

    config = get_configurable_parameters(model_name=args.model, config_path=args.config)
    if config.project.get("seed") is not None:
        seed_everything(config.project.seed)

    config = update_ensemble_input_size_config(config)
    tiler = EnsembleTiler(
        tile_size=config.dataset.tiling.tile_size,
        stride=config.dataset.tiling.stride,
        image_size=config.dataset.image_size,
        remove_border_count=config.dataset.tiling.remove_border_count,
    )

    tile_predictions = {}

    # go over all tile positions and train
    for tile_index in product(range(tiler.num_patches_h), range(tiler.num_patches_w)):
        logger.info(f"Start of procedure for tile {tile_index}")
        datamodule = get_datamodule(config)

        datamodule.custom_collate_fn = TileCollater(tiler, tile_index)

        model = get_model(config)
        experiment_logger = get_experiment_logger(config)
        callbacks = get_callbacks(config)

        trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=callbacks)
        logger.info("Training the model.")
        trainer.fit(model=model, datamodule=datamodule)

        predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path="best")
        tile_predictions[tile_index] = predictions

        if config.dataset.test_split_mode == TestSplitMode.NONE:
            logger.info("No test set provided. Skipping test stage.")
        else:
            logger.info("Testing the model.")
            # trainer.test(model=model, datamodule=datamodule)

    joiner = BasicPredictionJoiner(tile_predictions, tiler)

    all_predictions = joiner.join_tile_predictions()

    logger.info("Visualizing the results.")
    visualize_results(all_predictions, config)


if __name__ == "__main__":
    args = get_parser().parse_args()
    train(args)
