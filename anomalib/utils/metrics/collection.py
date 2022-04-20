"""Anomalib Metric Collection."""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from torchmetrics import MetricCollection


class AnomalibMetricCollection(MetricCollection):
    """Extends the MetricCollection class for use in the Anomalib pipeline."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._update_called = False
        self._threshold = 0.5

    def set_threshold(self, threshold_value):
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
