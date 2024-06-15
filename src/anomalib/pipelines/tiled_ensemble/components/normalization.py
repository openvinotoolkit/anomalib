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
        predictions (list[Any]): list of predictions.
        root_dir (Path): Root directory to save checkpoints, stats and images.
    """

    name = "pipeline"

    def __init__(self, predictions: list, root_dir: Path) -> None:
        super().__init__()
        self.predictions = predictions
        self.root_dir = root_dir

    def run(self, task_id: int | None = None) -> list[Any]:
        """Run normalization job which normalizes image, pixel and box scores.

        Args:
            task_id: not used in this case

        Returns:
            list[Any]: list of normalized predictions.
        """
        del task_id  # not needed here

        stats_path = self.root_dir / "weights" / "lightning" / "stats.json"
        with stats_path.open("r") as f:
            stats = json.load(f)
        min_val = stats["min"]
        max_val = stats["max"]
        image_threshold = stats["image_threshold"]
        pixel_threshold = stats["pixel_threshold"]

        logger.info("Starting normalization.")

        for data in tqdm(self.predictions, desc="Normalizing"):
            data["pred_scores"] = normalize(data["pred_scores"], image_threshold, min_val, max_val)
            if "anomaly_maps" in data:
                data["anomaly_maps"] = normalize(data["anomaly_maps"], pixel_threshold, min_val, max_val)
            if "box_scores" in data:
                data["box_scores"] = [
                    normalize(scores, pixel_threshold, min_val, max_val) for scores in data["box_scores"]
                ]

        return self.predictions

    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            list[Any]: list of predictions.
        """
        # take the first element as result is list of lists here
        return results[0]

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """Nothing is saved in this job."""


class NormalizationJobGenerator(JobGenerator):
    """Generate NormalizationJob.

    Args:
        root_dir (Path): Root directory to save checkpoints, stats and images.
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
    ) -> Generator[Job, None, None]:
        """Return a generator producing a single normalization job.

        Args:
            args: not used here.
            prev_stage_result (list[Any]): ensemble predictions from previous step.

        Returns:
            Generator[Job, None, None]: NormalizationJob generator
        """
        del args  # not needed here

        yield NormalizationJob(prev_stage_result, self.root_dir)
