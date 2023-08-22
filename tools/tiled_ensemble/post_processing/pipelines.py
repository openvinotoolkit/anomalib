"""Functions used to obtain and execute ensemble post-processing pipelines."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

from omegaconf import DictConfig, ListConfig
from tools.tiled_ensemble.ensemble_tiler import EnsembleTiler
from tools.tiled_ensemble.post_processing.metrics import EnsembleMetrics
from tools.tiled_ensemble.post_processing.postprocess import (
    EnsemblePostProcess,
    EnsemblePostProcessPipeline,
    MinMaxNormalize,
    NormalizationStage,
    PostProcessStats,
    SmoothJoins,
    Threshold,
)
from tools.tiled_ensemble.post_processing.visualization import EnsembleVisualization
from tools.tiled_ensemble.predictions import EnsemblePredictionJoiner, EnsemblePredictions

from anomalib.data.utils import TestSplitMode
from anomalib.post_processing import ThresholdMethod

logger = logging.getLogger(__name__)


def get_stats_pipeline(config: DictConfig | ListConfig, tiler: EnsembleTiler) -> EnsemblePostProcessPipeline:
    """
    Construct pipeline used to obtain prediction statistics, like min & max and image & pixel thresholds.

    Stats are usually obtained from validation set.

    Args:
        config (DictConfig | ListConfig): Configurable parameters object.
        tiler (EnsembleTiler): Tiler used by some steps of pipeline.

    Returns:
        EnsemblePostProcessPipeline: Constructed pipeline.
    """
    stats_pipeline = EnsemblePostProcessPipeline(EnsemblePredictionJoiner(tiler))

    steps: list[EnsemblePostProcess] = []

    if config.ensemble.post_processing.smooth_joins.apply:
        smooth_joins = SmoothJoins(
            width_factor=config.ensemble.post_processing.smooth_joins.width,
            filter_sigma=config.ensemble.post_processing.smooth_joins.sigma,
            tiler=tiler,
        )
        steps.append(smooth_joins)
    if (
        config.ensemble.metrics.threshold.method == ThresholdMethod.ADAPTIVE
        or config.ensemble.post_processing.normalization == NormalizationStage.JOINED_IMAGE
    ):
        steps.append(PostProcessStats())

    stats_pipeline.add_steps(steps)

    return stats_pipeline


def get_stats(
    config: DictConfig | ListConfig, tiler: EnsembleTiler, validation_predictions: EnsemblePredictions | None
) -> dict:
    """
    Get statistics used for postprocessing. This includes min & max and image & pixel thresholds.

    If validation data is none, then defaults are retunred:
    {"min": float("inf"), "max": float("-inf"), "image_threshold": 0.5, "pixel_threshold": 0.5}

    Args:
        config (DictConfig | ListConfig): Configurable parameters object.
        tiler (EnsembleTiler): Tiler used by some steps of pipeline.
        validation_predictions (EnsemblePredictions | None): Predictions used to calculate stats.

    Returns:
        dict: Dictionary with calculated statistics.

    Example:
        >>> from tools.tiled_ensemble.ensemble_tiler import EnsembleTiler
        >>> from tools.tiled_ensemble.predictions.prediction_data import BasicEnsemblePredictions
        >>>
        >>> tiler = EnsembleTiler(tile_size=256, stride=128, image_size=512)
        >>> # this data should be from validation dataloader predictions, as stats are usually obtained on validation.
        >>> data = BasicEnsemblePredictions()
        >>>
        >>> stats = get_stats(config, tiler, data)
        >>> stats
        {"min": 0, "max": 1, "image_threshold": 0.42, "pixel_threshold": 0.13,}
    """
    # if no validation set was provided
    if validation_predictions is None:
        warnings.warn("No validation set provided. Proceeding with default post-processing statistics.")
        return {"min": float("inf"), "max": float("-inf"), "image_threshold": 0.5, "pixel_threshold": 0.5}

    stats_pipeline = get_stats_pipeline(config, tiler)

    pipe_out = stats_pipeline.execute(validation_predictions)

    return pipe_out.get("stats", {})


def log_postprocess_steps(steps: list[EnsemblePostProcess]) -> None:
    """
    Log steps used in post-processing pipeline.

    Args:
        steps (list[EnsemblePostProcess]): List of steps in pipeline.

    """
    logger.info("-" * 42)
    logger.info("Steps in post processing pipeline:")
    for step in steps:
        logger.info(step.name)
    logger.info("-" * 42)


def get_postprocessing_pipeline(
    config: DictConfig | ListConfig, tiler: EnsembleTiler, stats: dict
) -> EnsemblePostProcessPipeline:
    """
    Construct pipeline used to post process ensemble predictions.

    Blocks of pipeline are determined based on ensemble config file since some steps can be done tile wise,
    or on joined images. Visualization and metrics are also configured here.

    Args:
        config (DictConfig | ListConfig): Configurable parameters object.
        tiler (EnsembleTiler): Tiler used by some steps of pipeline.
        stats (dict): Statistics of predictions (min, max, thresholds).

    Returns:
        EnsemblePostProcessPipeline: Constructed pipeline.
    """
    post_pipeline = EnsemblePostProcessPipeline(EnsemblePredictionJoiner(tiler))

    steps: list[EnsemblePostProcess] = []
    if config.ensemble.post_processing.smooth_joins.apply:
        smooth_joins = SmoothJoins(
            width_factor=config.ensemble.post_processing.smooth_joins.width,
            filter_sigma=config.ensemble.post_processing.smooth_joins.sigma,
            tiler=tiler,
        )
        steps.append(smooth_joins)

    # override threshold if it's set manually
    if config.ensemble.metrics.threshold.method == ThresholdMethod.MANUAL:
        stats["image_threshold"] = config.ensemble.metrics.threshold.manual_image
        stats["pixel_threshold"] = config.ensemble.metrics.threshold.manual_pixel

    # if normalization is done at the end on image-level
    if config.ensemble.post_processing.normalization == NormalizationStage.JOINED_IMAGE:
        steps.append(MinMaxNormalize(stats))
        # with minmax normalization, values are normalized such that the threshold value is centered at 0.5
        stats["image_threshold"] = 0.5
        stats["pixel_threshold"] = 0.5

    # if thresholding is done at the end on image-level
    if config.ensemble.metrics.threshold.stage == NormalizationStage.JOINED_IMAGE:
        steps.append(Threshold(stats["image_threshold"], stats["pixel_threshold"]))

    if config.ensemble.visualization.show_images or config.ensemble.visualization.save_images:
        visualizer = EnsembleVisualization(
            mode=config.ensemble.visualization.mode,
            task=config.dataset.task,
            save_images=config.ensemble.visualization.save_images,
            show_images=config.ensemble.visualization.show_images,
            save_path=config.ensemble.visualization.image_save_path or config.project.path + "/images",
        )
        steps.append(visualizer)

    steps.append(
        EnsembleMetrics(
            config.dataset.task,
            config.ensemble.metrics.get("image", None),
            config.ensemble.metrics.get("pixel", None),
            stats["image_threshold"],
            stats["pixel_threshold"],
        )
    )
    post_pipeline.add_steps(steps)

    log_postprocess_steps(steps)

    return post_pipeline


def post_process(
    config: DictConfig | ListConfig,
    tiler: EnsembleTiler,
    ensemble_predictions: EnsemblePredictions | None,
    validation_predictions: EnsemblePredictions | None,
    stats: dict | None = None,
) -> dict:
    """
    Postprocess, visualize and calculate metrics.

    Steps in postprocessing are determined from ensemble config file.
    Check `get_postprocessing_pipeline` for more details.

    Args:
        config (DictConfig | ListConfig): Configurable parameters object.
        tiler (EnsembleTiler): Tiler used for untiling of predictions.
        ensemble_predictions (EnsemblePredictions | None): Predictions to be joined and processed.
        validation_predictions (EnsemblePredictions | None): Predictions used to calculate stats.
        stats (dict | None): Dictionary containing statistics used for postprocessing. If None, run stats pipeline.

    Returns:
        dict: Dictionary with calculated metrics data.

    Examples:
        >>> # example of post_processing in train run
        >>> from tools.tiled_ensemble.ensemble_functions import get_prediction_storage
        >>> pred, val_pred = get_prediction_storage(config)
        >>> # ... pred and val_pred are filled during training
        >>> # post-processing with steps based on config file
        >>> post_process(config, tiler, ensemble_predictions=pred, validation_predictions=val_pred)
        {'image_F1Score': 0.42, 'image_AUROC': 0.42, 'pixel_F1Score': 0.42, 'pixel_AUROC': 0.42}

        >>> # example of post_processing in test run
        >>> from tools.tiled_ensemble.ensemble_functions import get_prediction_storage
        >>> pred, _ = get_prediction_storage(config)
        >>> # ... pred is filled during test prediction
        >>> # stats are loaded from json
        >>> # post-processing with steps based on config file
        >>> post_process(config, tiler, ensemble_predictions=pred, validation_predictions=None, stats=stats)
        {'image_F1Score': 0.42, 'image_AUROC': 0.42, 'pixel_F1Score': 0.42, 'pixel_AUROC': 0.42}

    """
    if stats is None:
        logger.info("Computing normalization and threshold statistics.")
        # get statistics, calculated on validation dataset
        stats = get_stats(config, tiler, validation_predictions)

        # save stats to json for later use in inference
        logger.info("Saving statistics to project directory.")
        stats_path = Path(config.project.path) / "weights" / "lightning" / "stats.json"
        with open(stats_path, "w", encoding="utf-8") as stats_file:
            json.dump(stats, stats_file, ensure_ascii=False, indent=4)
    else:
        logger.info("Using the provided normalization and threshold statistics.")

    if config.dataset.test_split_mode == TestSplitMode.NONE:
        logger.info("No test set provided. Skipping post-processing of test data.")
        return {}

    post_pipeline = get_postprocessing_pipeline(config, tiler, stats)

    logger.info("Executing post-processing pipeline on test data.")
    # add all above configured steps to pipeline and execute
    pipe_out = post_pipeline.execute(ensemble_predictions)

    return pipe_out["metrics"]
