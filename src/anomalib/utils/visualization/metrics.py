"""Metrics visualization generator for anomaly detection results.

This module provides utilities for visualizing metric plots from anomaly detection
models. The key components include:

    - Automatic generation of metric plots from model metrics
    - Support for both image-level and pixel-level metrics
    - Consistent file naming and output format

Example:
    >>> from anomalib.utils.visualization import MetricsVisualizer
    >>> # Create metrics visualizer
    >>> visualizer = MetricsVisualizer()
    >>> # Generate metric plots
    >>> results = visualizer.generate(pl_module=model)

The module ensures proper visualization of model performance metrics by:
    - Automatically detecting plottable metrics
    - Generating standardized plot formats
    - Handling both classification and segmentation metrics
    - Providing consistent file naming conventions

Note:
    Metrics must implement a ``generate_figure`` method to be visualized.
    The method should return a tuple of (figure, log_name).
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from typing import TYPE_CHECKING

from .base import BaseVisualizer, GeneratorResult, VisualizationStep

if TYPE_CHECKING:
    from anomalib.models import AnomalibModule


class MetricsVisualizer(BaseVisualizer):
    """Generate metric plots from model metrics.

    This class handles the automatic generation of metric plots from an anomalib
    model's metrics. It supports both image-level and pixel-level metrics.
    """

    def __init__(self) -> None:
        super().__init__(VisualizationStep.STAGE_END)

    @staticmethod
    def generate(**kwargs) -> Iterator[GeneratorResult]:
        """Generate metric plots and return them as an iterator.

        Args:
            **kwargs: Keyword arguments passed to the generator.
                Must include ``pl_module`` containing the model metrics.

        Yields:
            Iterator[GeneratorResult]: Generator results containing the plot
                figures and filenames.

        Raises:
            ValueError: If ``pl_module`` is not provided in kwargs.

        Example:
            >>> visualizer = MetricsVisualizer()
            >>> for result in visualizer.generate(pl_module=model):
            ...     # Process the visualization result
            ...     print(result.file_name)
        """
        pl_module: AnomalibModule = kwargs.get("pl_module", None)
        if pl_module is None:
            msg = "`pl_module` must be provided"
            raise ValueError(msg)
        for metrics in (pl_module.image_metrics, pl_module.pixel_metrics):
            for metric in metrics.values():
                # `generate_figure` needs to be defined for every metric that should be plotted automatically
                if hasattr(metric, "generate_figure"):
                    fig, log_name = metric.generate_figure()
                    file_name = f"{metrics.prefix}{log_name}.png"
                    yield GeneratorResult(image=fig, file_name=file_name)
