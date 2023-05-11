"""Base class for thresholding metrics."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC
from typing import Any

from torchmetrics import Metric


class BaseAnomalyScoreThreshold(Metric, ABC):
    """Base class for thresholding metrics."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def compute(self) -> Any:
        """Compute the threshold
        Returns:
            Value of the optimal threshold.
        """
        raise NotImplementedError("Subclass of BaseAnomalyScoreThreshold must implement the compute method")

    def update(self, *args, **kwargs) -> None:
        """Update the metric state
        Args:
            *args: Any positional arguments.
            **kwargs: Any keyword arguments.
        """
        raise NotImplementedError("Subclass of BaseAnomalyScoreThreshold must implement the update method")

    # def state_dict(
    #     self, destination: dict[str, Any], prefix: str = "", keep_vars: bool = False
    # ) -> dict[str, Any] | None:
    #     """Since class name is added to the keys, this makes it easier to load the correct class from weights.
    #     Note: This is temporary and will be replaced when custom loops are merged.
    #     """
    #     prefix = f"{self.__class__.__name__}.{prefix}"
    #     return super().state_dict(destination, prefix, keep_vars)
