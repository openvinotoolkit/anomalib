"""Tiled ensemble - prediction merging job."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Generator
from typing import Any

from tqdm import tqdm

from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.types import GATHERED_RESULTS, RUN_RESULTS

from .utils.ensemble_tiling import EnsembleTiler
from .utils.helper_functions import get_ensemble_tiler
from .utils.prediction_data import EnsemblePredictions
from .utils.prediction_merging import PredictionMergingMechanism

logger = logging.getLogger(__name__)


class MergeJob(Job):
    """Job for merging tile-level predictions into image-level predictions.

    Args:
        predictions (EnsemblePredictions): Object containing ensemble predictions.
        tiler (EnsembleTiler): Ensemble tiler used for untiling.
    """

    name = "Merge"

    def __init__(self, predictions: EnsemblePredictions, tiler: EnsembleTiler) -> None:
        super().__init__()
        self.predictions = predictions
        self.tiler = tiler

    def run(self, task_id: int | None = None) -> list[Any]:
        """Run merging job that merges all batches of tile-level predictions into image-level predictions.

        Args:
            task_id: Not used in this case.

        Returns:
            list[Any]: List of merged predictions.
        """
        del task_id  # not needed here

        merger = PredictionMergingMechanism(self.predictions, self.tiler)

        logger.info("Merging predictions.")

        # merge all batches
        merged_predictions = [
            merger.merge_tile_predictions(batch_idx)
            for batch_idx in tqdm(range(merger.num_batches), desc="Prediction merging")
        ]

        return merged_predictions  # noqa: RET504

    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            list[Any]: List of predictions.
        """
        # take the first element as result is list of lists here
        return results[0]

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """Nothing to save in this job."""


class MergeJobGenerator(JobGenerator):
    """Generate MergeJob."""

    def __init__(self, tiling_args: dict, data_args: dict) -> None:
        super().__init__()
        self.tiling_args = tiling_args
        self.data_args = data_args

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return MergeJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: EnsemblePredictions | None = None,
    ) -> Generator[MergeJob, None, None]:
        """Return a generator producing a single merging job.

        Args:
            args (dict): Tiled ensemble pipeline args.
            prev_stage_result (EnsemblePredictions): Ensemble predictions from predict step.

        Returns:
            Generator[MergeJob, None, None]: MergeJob generator
        """
        del args  # args not used here

        tiler = get_ensemble_tiler(self.tiling_args, self.data_args)
        if prev_stage_result is not None:
            yield MergeJob(prev_stage_result, tiler)
        else:
            msg = "Merging job requires tile level predictions from previous step."
            raise ValueError(msg)
