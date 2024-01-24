"""Anomalib Metric Collection."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from torchmetrics import MetricCollection


class AnomalibMetricCollection(MetricCollection):
    """Extends the MetricCollection class for use in the Anomalib pipeline."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._update_called = False
        self._threshold = 0.5

    def set_threshold(self, threshold_value: float) -> None:
        """Update the threshold value for all metrics that have the threshold attribute."""
        self._threshold = threshold_value
        for metric in self.values():
            if hasattr(metric, "threshold"):
                metric.threshold = threshold_value

    def update(self, *args, **kwargs) -> None:
        """Add data to the metrics."""
        super().update(*args, **kwargs)
        self._update_called = True

    @property
    def update_called(self) -> bool:
        """Returns a boolean indicating if the update method has been called at least once."""
        return self._update_called

    @property
    def threshold(self) -> float:
        """Return the value of the anomaly threshold."""
        return self._threshold
