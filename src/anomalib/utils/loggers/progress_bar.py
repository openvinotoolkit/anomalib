"""Logger to store metrics for displaying in the progress bar."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace
from typing import Any, Dict

from lightning.pytorch.loggers import Logger


class ProgressBarMetricLogger(Logger):
    def __init__(self):
        super().__init__()
        self._metrics = {}

    def log_hyperparams(self, params: Dict[str, Any] | Namespace, *args: Any, **kwargs: Any) -> None:
        return

    @property
    def name(self) -> str | None:
        return "progress_bar"

    @property
    def version(self) -> str | None:
        return None

    def log_metrics(self, metrics: Dict[str, float], step: int | None = None) -> None:
        self._metrics.update(metrics)
