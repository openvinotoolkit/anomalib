"""Tiled ensemble - post-processing statistics calculation job."""
import json

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from anomalib.metrics import F1AdaptiveThreshold, MinMax
from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.types import GATHERED_RESULTS, RUN_RESULTS

logger = logging.getLogger(__name__)


class StatisticsJob(Job):
    """Job for calculating min, max and threshold statistics for post-processing.

    Args:
        predictions (list[Any]): list of image-level predictions.
        root_dir (Path): Root directory to save checkpoints, stats and images.
    """

    name = "pipeline"

    def __init__(self, predictions: list, root_dir: Path) -> None:
        super().__init__()
        self.predictions = predictions
        self.root_dir = root_dir

    def run(self, task_id: int | None = None) -> dict:
        """Run job that calculates statistics needed in post-processing steps.

        Args:
            task_id: not used in this case

        Returns:
            dict: statistics dict with min, max and threshold values.
        """
        del task_id  # not needed here

        minmax = MinMax().cpu()
        image_threshold = F1AdaptiveThreshold()
        pixel_threshold = F1AdaptiveThreshold()
        pixel_update_called = False

        logger.info("Starting post-processing statistics calculation.")

        for data in tqdm(self.predictions, desc="Stats calculation"):
            # update minmax
            if "anomaly_maps" in data:
                minmax(data["anomaly_maps"])
            elif "box_scores" in data:
                minmax(torch.cat(data["box_scores"]))
            elif "pred_scores" in data:
                minmax(data["pred_scores"])
            else:
                msg = "No values found for normalization, provide anomaly maps, bbox scores, or image scores"
                raise ValueError(msg)

            # update thresholds
            image_threshold.update(data["pred_scores"], data["label"].int())
            if "mask" in data and "anomaly_maps" in data:
                pixel_threshold.update(torch.squeeze(data["anomaly_maps"]), torch.squeeze(data["mask"].int()))
                pixel_update_called = True

        image_threshold.compute()
        if pixel_update_called:
            pixel_threshold.compute()
        else:
            pixel_threshold.value = image_threshold.value

        # return stats with save path that is later used to save statistics.
        return {
            "min": minmax.min.item(),
            "max": minmax.max.item(),
            "image_threshold": image_threshold.value.item(),
            "pixel_threshold": pixel_threshold.value.item(),
            "save_path": (self.root_dir / "weights" / "lightning" / "stats.json"),
        }

    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            dict: statistics dictionary.
        """
        # take the first element as result is list of lists here
        return results[0]

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """Save statistics to file system."""
        # get and remove path from stats dict
        stats_path: Path = results.pop("save_path")
        stats_path.parent.mkdir(parents=True, exist_ok=True)

        # save statistics next to weights
        with stats_path.open("w", encoding="utf-8") as stats_file:
            json.dump(results, stats_file, ensure_ascii=False, indent=4)


class StatisticsJobGenerator(JobGenerator):
    """Generate StatisticsJob.

    Args:
        root_dir (Path): Root directory to save checkpoints, stats and images.
    """

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return StatisticsJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: list[Any] | None = None,
    ) -> Generator[Job, None, None]:
        """Return a generator producing a single stats calculating job.

        Args:
            args: not used here.
            prev_stage_result (list[Any]): ensemble predictions from previous step.

        Returns:
            Generator[Job, None, None]: StatisticsJob generator
        """
        del args  # not needed here

        yield StatisticsJob(prev_stage_result, self.root_dir)
