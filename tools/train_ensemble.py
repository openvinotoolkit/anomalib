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
from tqdm import tqdm

from anomalib.config import get_configurable_parameters
from anomalib.data import get_datamodule
from anomalib.data.utils import ValSplitMode
from anomalib.models import get_model
from anomalib.models.ensemble.ensemble_postprocess import (
    PostProcessStats,
    EnsemblePostProcessPipeline,
    SmoothJoins,
    MinMaxNormalize,
    Threshold,
)
from anomalib.models.ensemble.ensemble_visualization import EnsembleVisualization
from anomalib.utils.callbacks import (
    get_callbacks,
    LoadModelCallback,
    ImageVisualizerCallback,
    MetricVisualizerCallback,
    MinMaxNormalizationCallback,
)
from anomalib.utils.loggers import configure_logger, get_experiment_logger

from anomalib.models.ensemble.ensemble_tiler import EnsembleTiler
from anomalib.models.ensemble.ensemble_functions import (
    TileCollater,
    update_ensemble_input_size_config,
    BasicPredictionJoiner,
)
from anomalib.models.ensemble.ensemble_prediction_data import (
    BasicEnsemblePredictions,
    FileSystemEnsemblePredictions,
    RescaledEnsemblePredictions,
)
from anomalib.models.ensemble.ensemble_metrics import EnsembleMetrics, log_metrics

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

    # TODO: refactor into ensemble suitable
    config = get_configurable_parameters(model_name=args.model, config_path=args.config)
    if config.project.get("seed") is not None:
        seed_everything(config.project.seed)

    config = update_ensemble_input_size_config(config)
    # TODO: refactor where it accepts config
    tiler = EnsembleTiler(
        tile_size=config.dataset.tiling.tile_size,
        stride=config.dataset.tiling.stride,
        image_size=config.dataset.image_size,
        remove_border_count=config.dataset.tiling.remove_border_count,
    )

    experiment_logger = get_experiment_logger(config)

    # prepare datamodule and set collate function that performs tiling
    # TODO: refactor into one function
    datamodule = get_datamodule(config)
    tile_collater = TileCollater(tiler, (0, 0))
    datamodule.custom_collate_fn = tile_collater

    ensemble_predictions = BasicEnsemblePredictions()
    if config.dataset.val_split_mode == ValSplitMode.SAME_AS_TEST:
        validation_predictions = None
    else:
        validation_predictions = BasicEnsemblePredictions()

    # go over all tile positions and train
    for tile_index in product(range(tiler.num_patches_h), range(tiler.num_patches_w)):
        logger.info(f"Start of procedure for tile {tile_index}")

        # TODO: refactor into separate function
        # configure callbacks for ensemble
        callbacks = get_callbacks(config)
        ensemble_callbacks = []
        # temporary removing for ensemble
        for callback in callbacks:
            if not isinstance(
                callback,
                (
                    ImageVisualizerCallback,
                    MetricVisualizerCallback,
                ),
            ):
                ensemble_callbacks.append(callback)

        # set tile position inside dataloader
        tile_collater.tile_index = tile_index

        model = get_model(config)

        trainer = Trainer(**config.trainer, logger=experiment_logger, callbacks=ensemble_callbacks.copy())
        logger.info("Training the model.")
        trainer.fit(model=model, datamodule=datamodule)

        logger.info("Loading the best model weights.")
        load_model_callback = LoadModelCallback(weights_path=trainer.checkpoint_callback.best_model_path)
        trainer.callbacks.insert(0, load_model_callback)  # pylint: disable=no-member

        current_predictions = trainer.predict(model=model, datamodule=datamodule)
        ensemble_predictions.add_tile_prediction(tile_index, current_predictions)

        if validation_predictions:
            current_val_predictions = trainer.predict(model=model, dataloaders=datamodule.val_dataloader())
            validation_predictions.add_tile_prediction(tile_index, current_val_predictions)

    joiner = BasicPredictionJoiner(tiler)

    if not validation_predictions:
        validation_predictions = ensemble_predictions

    # get normalization and threshold
    stats_pipeline = EnsemblePostProcessPipeline(validation_predictions, joiner)
    stats_pipeline.add_steps([SmoothJoins(), PostProcessStats()])
    stats = stats_pipeline.execute()["stats"]

    # build postprocess, visualization and metric pipeline
    post_pipeline = EnsemblePostProcessPipeline(ensemble_predictions, joiner)
    post_pipeline.add_steps(
        [
            SmoothJoins(),
            MinMaxNormalize(stats),
            Threshold(0.5, 0.5),
            EnsembleVisualization(config),
            EnsembleMetrics(config, 0.5, 0.5),
        ]
    )
    # execute pipeline and take metric results
    computed_metrics = post_pipeline.execute()["metrics"]

    log_metrics(computed_metrics)


if __name__ == "__main__":
    args = get_parser().parse_args()
    train(args)
