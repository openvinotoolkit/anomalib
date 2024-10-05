"""Tiled ensemble - metrics calculation job."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from anomalib import TaskType
from anomalib.metrics import AnomalibMetricCollection, create_metric_collection
from anomalib.pipelines.components import Job, JobGenerator
from anomalib.pipelines.types import GATHERED_RESULTS, PREV_STAGE_RESULT, RUN_RESULTS

from .utils import NormalizationStage
from .utils.helper_functions import get_threshold_values

logger = logging.getLogger(__name__)


class MetricsCalculationJob(Job):
    """Job for image and pixel metrics calculation.

    Args:
        accelerator (str): Accelerator (device) to use.
        predictions (list[Any]): List of batch predictions.
        root_dir (Path): Root directory to save checkpoints, stats and images.
        image_metrics (AnomalibMetricCollection): Collection of all image-level metrics.
        pixel_metrics (AnomalibMetricCollection): Collection of all pixel-level metrics.
    """

    name = "Metrics"

    def __init__(
        self,
        accelerator: str,
        predictions: list[Any] | None,
        root_dir: Path,
        image_metrics: AnomalibMetricCollection,
        pixel_metrics: AnomalibMetricCollection,
    ) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.predictions = predictions
        self.root_dir = root_dir
        self.image_metrics = image_metrics
        self.pixel_metrics = pixel_metrics

    def run(self, task_id: int | None = None) -> dict:
        """Run a job that calculates image and pixel level metrics.

        Args:
            task_id: Not used in this case.

        Returns:
            dict[str, float]: Dictionary containing calculated metric values.
        """
        del task_id  # not needed here

        logger.info("Starting metrics calculation.")

        # add predicted data to metrics
        for data in tqdm(self.predictions, desc="Calculating metrics"):
            self.image_metrics.update(data["pred_scores"], data["label"].int())
            if "mask" in data and "anomaly_maps" in data:
                self.pixel_metrics.update(data["anomaly_maps"], data["mask"].int())

        # compute all metrics on specified accelerator
        metrics_dict = {}
        for name, metric in self.image_metrics.items():
            metric.to(self.accelerator)
            metrics_dict[name] = metric.compute().item()
            metric.cpu()

        if self.pixel_metrics.update_called:
            for name, metric in self.pixel_metrics.items():
                metric.to(self.accelerator)
                metrics_dict[name] = metric.compute().item()
                metric.cpu()

        for name, value in metrics_dict.items():
            print(f"{name}: {value:.4f}")

        # save path used in `save` method
        metrics_dict["save_path"] = self.root_dir / "metric_results.csv"

        return metrics_dict

    @staticmethod
    def collect(results: list[RUN_RESULTS]) -> GATHERED_RESULTS:
        """Nothing to collect in this job.

        Returns:
            list[Any]: list of predictions.
        """
        # take the first element as result is list of dict here
        return results[0]

    @staticmethod
    def save(results: GATHERED_RESULTS) -> None:
        """Save metrics values to csv."""
        logger.info("Saving metrics to csv.")

        # get and remove path from stats dict
        results_path: Path = results.pop("save_path")
        results_path.parent.mkdir(parents=True, exist_ok=True)

        df_dict = {k: [v] for k, v in results.items()}
        metrics_df = pd.DataFrame(df_dict)
        metrics_df.to_csv(results_path, index=False)


class MetricsCalculationJobGenerator(JobGenerator):
    """Generate MetricsCalculationJob.

    Args:
        root_dir (Path): Root directory to save checkpoints, stats and images.
    """

    def __init__(
        self,
        accelerator: str,
        root_dir: Path,
        task: TaskType,
        metrics: dict,
        normalization_stage: NormalizationStage,
    ) -> None:
        self.accelerator = accelerator
        self.root_dir = root_dir
        self.task = task
        self.metrics = metrics
        self.normalization_stage = normalization_stage

    @property
    def job_class(self) -> type:
        """Return the job class."""
        return MetricsCalculationJob

    def configure_ensemble_metrics(
        self,
        image_metrics: list[str] | dict[str, dict[str, Any]] | None = None,
        pixel_metrics: list[str] | dict[str, dict[str, Any]] | None = None,
    ) -> tuple[AnomalibMetricCollection, AnomalibMetricCollection]:
        """Configure image and pixel metrics and put them into a collection.

        Args:
            image_metrics (list[str] | None): List of image-level metric names.
            pixel_metrics (list[str] | None): List of pixel-level metric names.

        Returns:
            tuple[AnomalibMetricCollection, AnomalibMetricCollection]:
                Image-metrics collection and pixel-metrics collection
        """
        image_metrics = [] if image_metrics is None else image_metrics

        if pixel_metrics is None:
            pixel_metrics = []
        elif self.task == TaskType.CLASSIFICATION:
            pixel_metrics = []
            logger.warning(
                "Cannot perform pixel-level evaluation when task type is classification. "
                "Ignoring the following pixel-level metrics: %s",
                pixel_metrics,
            )

        # if a single metric is passed, transform to list to fit the creation function
        if isinstance(image_metrics, str):
            image_metrics = [image_metrics]
        if isinstance(pixel_metrics, str):
            pixel_metrics = [pixel_metrics]

        image_metrics_collection = create_metric_collection(image_metrics, "image_")
        pixel_metrics_collection = create_metric_collection(pixel_metrics, "pixel_")

        return image_metrics_collection, pixel_metrics_collection

    def generate_jobs(
        self,
        args: dict | None = None,
        prev_stage_result: PREV_STAGE_RESULT = None,
    ) -> Generator[MetricsCalculationJob, None, None]:
        """Make a generator that yields a single metrics calculation job.

        Args:
            args: ensemble run config.
            prev_stage_result: ensemble predictions from previous step.

        Returns:
            Generator[MetricsCalculationJob, None, None]: MetricsCalculationJob generator
        """
        del args  # args not used here

        image_metrics_config = self.metrics.get("image", None)
        pixel_metrics_config = self.metrics.get("pixel", None)

        image_threshold, pixel_threshold = get_threshold_values(self.normalization_stage, self.root_dir)

        image_metrics, pixel_metrics = self.configure_ensemble_metrics(
            image_metrics=image_metrics_config,
            pixel_metrics=pixel_metrics_config,
        )

        # set thresholds for metrics that need it
        image_metrics.set_threshold(image_threshold)
        pixel_metrics.set_threshold(pixel_threshold)

        yield MetricsCalculationJob(
            accelerator=self.accelerator,
            predictions=prev_stage_result,
            root_dir=self.root_dir,
            image_metrics=image_metrics,
            pixel_metrics=pixel_metrics,
        )
