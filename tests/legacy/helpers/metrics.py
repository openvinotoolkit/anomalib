"""Helpers for metrics tests."""

from typing import Tuple, Union

from omegaconf import DictConfig, ListConfig

from anomalib.utils.metrics import (
    AnomalibMetricCollection,
    metric_collection_from_names,
)


def get_metrics(config: Union[ListConfig, DictConfig]) -> Tuple[AnomalibMetricCollection, AnomalibMetricCollection]:
    """Create metric collections based on the config.

    Args:
        config (Union[DictConfig, ListConfig]): Config.yaml loaded using OmegaConf

    Returns:
        AnomalibMetricCollection: Image-level metric collection
        AnomalibMetricCollection: Pixel-level metric collection
    """
    image_metric_names = config.metrics.image if "image" in config.metrics.keys() else []
    pixel_metric_names = config.metrics.pixel if "pixel" in config.metrics.keys() else []
    image_metrics = metric_collection_from_names(image_metric_names, "image_")
    pixel_metrics = metric_collection_from_names(pixel_metric_names, "pixel_")
    return image_metrics, pixel_metrics
