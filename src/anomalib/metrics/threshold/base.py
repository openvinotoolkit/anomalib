"""Base class for thresholding metrics."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC

import torch
from torchmetrics import Metric


class BaseThreshold(Metric, ABC):
    """Base class for thresholding metrics."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def compute(self) -> torch.Tensor:
        """Compute the threshold.

        Returns:
            Value of the optimal threshold.
        """
        msg = "Subclass of BaseAnomalyScoreThreshold must implement the compute method"
        raise NotImplementedError(msg)

    def update(self, *args, **kwargs) -> None:  # noqa: ARG002
        """Update the metric state.

        Args:
            *args: Any positional arguments.
            **kwargs: Any keyword arguments.
        """
        msg = "Subclass of BaseAnomalyScoreThreshold must implement the update method"
        raise NotImplementedError(msg)
