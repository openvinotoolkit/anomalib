"""Base class for thresholding metrics."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import warnings

import torch
from torchmetrics import Metric


class Threshold(Metric):
    """Base class for thresholding metrics.

    This class serves as the foundation for all threshold-based metrics in the system.
    It inherits from torchmetrics.Metric and provides a common interface for
    threshold computation and updates.

    Subclasses should implement the `compute` and `update` methods to define
    specific threshold calculation logic.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def compute(self) -> torch.Tensor:  # noqa: PLR6301
        """Compute the threshold.

        Returns:
            Value of the optimal threshold.
        """
        msg = "Subclass of Threshold must implement the compute method"
        raise NotImplementedError(msg)

    def update(self, *args, **kwargs) -> None:  # noqa: ARG002, PLR6301
        """Update the metric state.

        Args:
            *args: Any positional arguments.
            **kwargs: Any keyword arguments.
        """
        msg = "Subclass of Threshold must implement the update method"
        raise NotImplementedError(msg)


class BaseThreshold(Threshold):
    """Alias for Threshold class for backward compatibility."""

    def __init__(self, **kwargs) -> None:
        warnings.warn(
            "BaseThreshold is deprecated and will be removed in a future version. Use Threshold instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)
