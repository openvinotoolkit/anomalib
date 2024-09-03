"""Metrics visualization generator."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator
from typing import TYPE_CHECKING

from .base import BaseVisualizer, GeneratorResult, VisualizationStep

if TYPE_CHECKING:
    from anomalib.models import AnomalyModule


class MetricsVisualizer(BaseVisualizer):
    """Generate metric plots."""

    def __init__(self) -> None:
        super().__init__(VisualizationStep.STAGE_END)

    @staticmethod
    def generate(**kwargs) -> Iterator[GeneratorResult]:
        """Generate metric plots and return them as an iterator."""
        pl_module: AnomalyModule = kwargs.get("pl_module", None)
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
