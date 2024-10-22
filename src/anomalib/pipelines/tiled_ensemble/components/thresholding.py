"""Tiled ensemble - thresholding job."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

from tqdm import tqdm

from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.types import GATHERED_RESULTS, RUN_RESULTS

from .utils import NormalizationStage
from .utils.helper_functions import get_threshold_values

logger = logging.getLogger(__name__)


class ThresholdingJob(Job):
    """Job used to threshold predictions, producing labels from scores.

    Args:
        predictions (list[Any]): List of predictions.
        image_threshold (float): Threshold used for image-level thresholding.
        pixel_threshold (float): Threshold used for pixel-level thresholding.
    """

    name = "Threshold"

    def __init__(self, predictions: list[Any] | None, image_threshold: float, pixel_threshold: float) -> None:
        super().__init__()
        self.predictions = predictions
        self.image_threshold = image_threshold
        self.pixel_threshold = pixel_threshold

    def run(self, task_id: int | None = None) -> list[Any] | None:
        """Run job that produces prediction labels from scores.

        Args:
            task_id: Not used in this case.

        Returns:
            list[Any]: List of thresholded predictions.
        """
        del task_id  # not needed here

        logger.info("Starting thresholding.")

        for data in tqdm(self.predictions, desc="Thresholding"):
            if "pred_scores" in data:
                data["pred_labels"] = data["pred_scores"] >= self.image_threshold
            if "anomaly_maps" in data:
                data["pred_masks"] = data["anomaly_maps"] >= self.pixel_threshold

        return self.predictions

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
        """Nothing is saved in this job."""


class ThresholdingJobGenerator(JobGenerator):
    """Generate ThresholdingJob.

    Args:
        root_dir (Path): Root directory containing post-processing stats.
    """

    def __init__(self, root_dir: Path, normalization_stage: NormalizationStage) -> None:
        self.root_dir = root_dir
        self.normalization_stage = normalization_stage

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return ThresholdingJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: list[Any] | None = None,
    ) -> Generator[ThresholdingJob, None, None]:
        """Return a generator producing a single thresholding job.

        Args:
            args: ensemble run args.
            prev_stage_result (list[Any]): Ensemble predictions from previous step.

        Returns:
            Generator[ThresholdingJob, None, None]: ThresholdingJob generator.
        """
        del args  # args not used here

        # get threshold values base on normalization
        image_threshold, pixel_threshold = get_threshold_values(self.normalization_stage, self.root_dir)

        yield ThresholdingJob(
            predictions=prev_stage_result,
            image_threshold=image_threshold,
            pixel_threshold=pixel_threshold,
        )
