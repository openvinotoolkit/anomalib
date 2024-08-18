"""Tiled ensemble - post-processing statistics calculation job."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

import torch
from torchmetrics import MetricCollection
from tqdm import tqdm

from anomalib.metrics import F1AdaptiveThreshold, MinMax
from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.types import GATHERED_RESULTS, RUN_RESULTS

logger = logging.getLogger(__name__)


class StatisticsJob(Job):
    """Job for calculating min, max and threshold statistics for post-processing.

    Args:
        predictions (list[Any]): List of image-level predictions.
        root_dir (Path): Root directory to save checkpoints, stats and images.
    """

    name = "Stats"

    def __init__(self, predictions: list[Any] | None, root_dir: Path) -> None:
        super().__init__()
        self.predictions = predictions
        self.root_dir = root_dir

    def run(self, task_id: int | None = None) -> dict:
        """Run job that calculates statistics needed in post-processing steps.

        Args:
            task_id: Not used in this case

        Returns:
            dict: Statistics dict with min, max and threshold values.
        """
        del task_id  # not needed here

        minmax = MetricCollection(
            {
                "anomaly_maps": MinMax().cpu(),
                "box_scores": MinMax().cpu(),
                "pred_scores": MinMax().cpu(),
            },
        )
        image_threshold = F1AdaptiveThreshold()
        pixel_threshold = F1AdaptiveThreshold()
        pixel_update_called = False

        logger.info("Starting post-processing statistics calculation.")

        for data in tqdm(self.predictions, desc="Stats calculation"):
            # update minmax
            if "anomaly_maps" in data:
                minmax["anomaly_maps"](data["anomaly_maps"])
            if "box_scores" in data:
                minmax["box_scores"](torch.cat(data["box_scores"]))
            if "pred_scores" in data:
                minmax["pred_scores"](data["pred_scores"])

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

        min_max_vals = {}
        for pred_name, pred_metric in minmax.items():
            min_max_vals[pred_name] = {
                "min": pred_metric.min.item(),
                "max": pred_metric.max.item(),
            }

        # return stats with save path that is later used to save statistics.
        return {
            "minmax": min_max_vals,
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
        root_dir (Path): Root directory where statistics file will be saved (in weights folder).
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
            args: Not used here.
            prev_stage_result (list[Any]): Ensemble predictions from previous step.

        Returns:
            Generator[Job, None, None]: StatisticsJob generator.
        """
        del args  # not needed here

        yield StatisticsJob(prev_stage_result, self.root_dir)
