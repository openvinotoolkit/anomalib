"""Tiled ensemble - post-processing statistics calculation job."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig
from torchmetrics import MetricCollection
from tqdm import tqdm

from anomalib.callbacks.thresholding import _ThresholdCallback
from anomalib.metrics import MinMax
from anomalib.metrics.threshold import Threshold
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

    def __init__(
        self,
        predictions: list[Any] | None,
        root_dir: Path,
        image_threshold: Threshold,
        pixel_threshold: Threshold,
    ) -> None:
        super().__init__()
        self.predictions = predictions
        self.root_dir = root_dir
        self.image_threshold = image_threshold
        self.pixel_threshold = pixel_threshold

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
                "pred_scores": MinMax().cpu(),
            },
        )
        pixel_update_called = False

        logger.info("Starting post-processing statistics calculation.")

        for data in tqdm(self.predictions, desc="Stats calculation"):
            # update minmax
            if "anomaly_maps" in data:
                minmax["anomaly_maps"](data["anomaly_maps"])
            if "pred_scores" in data:
                minmax["pred_scores"](data["pred_scores"])

            # update thresholds
            self.image_threshold.update(data["pred_scores"], data["label"].int())
            if "mask" in data and "anomaly_maps" in data:
                self.pixel_threshold.update(torch.squeeze(data["anomaly_maps"]), torch.squeeze(data["mask"].int()))
                pixel_update_called = True

        self.image_threshold.compute()
        if pixel_update_called:
            self.pixel_threshold.compute()
        else:
            self.pixel_threshold.value = self.image_threshold.value

        min_max_vals = {}
        for pred_name, pred_metric in minmax.items():
            min_max_vals[pred_name] = {
                "min": pred_metric.min.item(),
                "max": pred_metric.max.item(),
            }

        # return stats with save path that is later used to save statistics.
        return {
            "minmax": min_max_vals,
            "image_threshold": self.image_threshold.value.item(),
            "pixel_threshold": self.pixel_threshold.value.item(),
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

    def __init__(
        self,
        root_dir: Path,
        thresholding_method: DictConfig | str | ListConfig | list[dict[str, str | float]],
    ) -> None:
        self.root_dir = root_dir
        self.threshold = thresholding_method

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return StatisticsJob

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: list[Any] | None = None,
    ) -> Generator[StatisticsJob, None, None]:
        """Return a generator producing a single stats calculating job.

        Args:
            args: Not used here.
            prev_stage_result (list[Any]): Ensemble predictions from previous step.

        Returns:
            Generator[StatisticsJob, None, None]: StatisticsJob generator.
        """
        del args  # not needed here

        # get threshold class based config
        if isinstance(self.threshold, str | DictConfig):
            # single method provided
            image_threshold = _ThresholdCallback._get_threshold_from_config(self.threshold)  # noqa: SLF001
            pixel_threshold = image_threshold.clone()
        elif isinstance(self.threshold, ListConfig | list):
            # image and pixel method specified separately
            image_threshold = _ThresholdCallback._get_threshold_from_config(self.threshold[0])  # noqa: SLF001
            pixel_threshold = _ThresholdCallback._get_threshold_from_config(self.threshold[1])  # noqa: SLF001
        else:
            msg = f"Invalid threshold config {self.threshold}"
            raise TypeError(msg)

        yield StatisticsJob(
            predictions=prev_stage_result,
            root_dir=self.root_dir,
            image_threshold=image_threshold,
            pixel_threshold=pixel_threshold,
        )
