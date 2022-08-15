"""Base class for anomaly thresholding metrics."""

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch import Tensor
from torchmetrics import Metric


class BaseThreshold(Metric, ABC):
    """Base class for thresholding metrics."""

    def __init__(self, default_value: Optional[float] = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_state("value", default=torch.tensor(default_value), persistent=True)  # pylint: disable=not-callable
        self.value = torch.tensor(default_value)  # pylint: disable=not-callable

    # pylint: disable=arguments-differ
    @abstractmethod
    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update the metric.

        This provides a common update interface for all the thresholding metrics.
        """
        raise NotImplementedError()

    @abstractmethod
    def compute(self) -> Tensor:
        """Compute the threshold."""
        raise NotImplementedError()
