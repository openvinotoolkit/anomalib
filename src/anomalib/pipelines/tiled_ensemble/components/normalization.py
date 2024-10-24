"""Tiled ensemble - normalization job."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

from tqdm import tqdm

from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.types import GATHERED_RESULTS, RUN_RESULTS
from anomalib.utils.normalization.min_max import normalize

logger = logging.getLogger(__name__)


class NormalizationJob(Job):
    """Job for normalization of predictions.

    Args:
        predictions (list[Any]): List of predictions.
        root_dir (Path): Root directory containing statistics needed for normalization.
    """

    name = "Normalize"

    def __init__(self, predictions: list[Any] | None, root_dir: Path) -> None:
        super().__init__()
        self.predictions = predictions
        self.root_dir = root_dir

    def run(self, task_id: int | None = None) -> list[Any] | None:
        """Run normalization job which normalizes image, pixel and box scores.

        Args:
            task_id: Not used in this case.

        Returns:
            list[Any]: List of normalized predictions.
        """
        del task_id  # not needed here

        # load all statistics needed for normalization
        stats_path = self.root_dir / "weights" / "lightning" / "stats.json"
        with stats_path.open("r") as f:
            stats = json.load(f)
        minmax = stats["minmax"]
        image_threshold = stats["image_threshold"]
        pixel_threshold = stats["pixel_threshold"]

        logger.info("Starting normalization.")

        for data in tqdm(self.predictions, desc="Normalizing"):
            data["pred_scores"] = normalize(
                data["pred_scores"],
                image_threshold,
                minmax["pred_scores"]["min"],
                minmax["pred_scores"]["max"],
            )
            if "anomaly_maps" in data:
                data["anomaly_maps"] = normalize(
                    data["anomaly_maps"],
                    pixel_threshold,
                    minmax["anomaly_maps"]["min"],
                    minmax["anomaly_maps"]["max"],
                )

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


class NormalizationJobGenerator(JobGenerator):
    """Generate NormalizationJob.

    Args:
        root_dir (Path): Root directory where statistics are saved.
    """

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return NormalizationJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: list[Any] | None = None,
    ) -> Generator[NormalizationJob, None, None]:
        """Return a generator producing a single normalization job.

        Args:
            args: not used here.
            prev_stage_result (list[Any]): Ensemble predictions from previous step.

        Returns:
            Generator[NormalizationJob, None, None]: NormalizationJob generator.
        """
        del args  # not needed here

        yield NormalizationJob(prev_stage_result, self.root_dir)
