"""Anomalib Metric Collection."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from torchmetrics import MetricCollection
from typing import Sequence
from anomalib.data import Batch


class MetricWrapper(MetricCollection):
    """Extends the MetricCollection class for use in the Anomalib pipeline."""

    def __init__(self, field_names: Sequence[str], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._update_called = False
        self.field_names = field_names

    def update(self, batch: Batch, *args, **kwargs):
        values = [getattr(batch, key) for key in self.field_names]
        super().update(*values, *args, **kwargs)
        self._update_called = True

    @property
    def update_called(self) -> bool:
        """Returns a boolean indicating if the update method has been called at least once."""
        return self._update_called
