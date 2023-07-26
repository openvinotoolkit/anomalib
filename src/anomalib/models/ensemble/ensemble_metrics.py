"""Class used to compute metrics for ensemble predictions."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from omegaconf import DictConfig, ListConfig

from anomalib.data import TaskType
from anomalib.models.ensemble.ensemble_postprocess import EnsemblePostProcess
from anomalib.utils.metrics import AnomalibMetricCollection, create_metric_collection

logger = logging.getLogger(__name__)


class EnsembleMetrics(EnsemblePostProcess):
    """
    Args:
        config: Configurable parameters object.
        image_threshold: Threshold used for image metrics.
        pixel_threshold: Threshold used for pixel metrics.
    """

    def __init__(self, config: DictConfig | ListConfig, image_threshold: float, pixel_threshold: float):
        super().__init__(final_compute=True, name="metrics")

        self.image_metrics, self.pixel_metrics = self.configure_ensemble_metrics(
            config.dataset.task,
            config.metrics.get("image", None),
            config.metrics.get("pixel", None),
        )

        # set threshold for metrics that require it
        self.image_metrics.set_threshold(image_threshold)
        self.pixel_metrics.set_threshold(pixel_threshold)

        self.image_metrics.cpu()
        self.pixel_metrics.cpu()

    def configure_ensemble_metrics(
        self,
        task: TaskType = TaskType.SEGMENTATION,
        image_metric_names: list[str] | None = None,
        pixel_metric_names: list[str] | None = None,
    ) -> (AnomalibMetricCollection, AnomalibMetricCollection):
        """
        Configure image and pixel metrics and put them into a collection.
        Args:
            task: Task type of the current run.
            image_metric_names: List of image-level metric names.
            pixel_metric_names: List of pixel-level metric names.

        Returns:
            image metrics collection and pixel metrics collection
        """
        image_metric_names = [] if image_metric_names is None else image_metric_names

        pixel_metric_names: list[str]
        if pixel_metric_names is None:
            pixel_metric_names = []
        elif task == TaskType.CLASSIFICATION:
            pixel_metric_names = []
            logger.warning(
                "Cannot perform pixel-level evaluation when task type is classification. "
                "Ignoring the following pixel-level metrics: %s",
                pixel_metric_names,
            )
        else:
            pixel_metric_names = pixel_metric_names

        image_metrics = create_metric_collection(image_metric_names, "image_")
        pixel_metrics = create_metric_collection(pixel_metric_names, "pixel_")

        return image_metrics, pixel_metrics

    def process(self, batch_results: dict) -> dict:
        """
        Compute metrics specified in config for given ensemble results.

        Args:
            batch_results: Joined batch of results produced by model ensemble.
        """

        self.image_metrics.update(batch_results["pred_scores"], batch_results["label"].int())
        if "mask" in batch_results.keys() and "anomaly_maps" in batch_results.keys():
            self.pixel_metrics.update(batch_results["anomaly_maps"], batch_results["mask"].int())

        return batch_results

    def compute(self) -> Any:
        """
        Compute metrics for entire ensemble.

        Returns:
            Dictionary containing calculated metric data.
        """
        out = {}
        for name, metric in self.image_metrics.items():
            out[name] = metric.compute().item()

        if self.pixel_metrics.update_called:
            for name, metric in self.pixel_metrics.items():
                out[name] = metric.compute().item()

        return out


def log_metrics(metric_dict: dict[str, float]) -> None:
    """
    Log computed metrics.

    Args:
        metric_dict: Dictionary containing all metrics info.
    """
    for name, value in metric_dict.items():
        logger.info(f"{name}: {value}")
