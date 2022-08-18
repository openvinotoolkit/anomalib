"""Class to store values of manual thresholds."""


from torch import Tensor

from .base import BaseThreshold


class ManualThreshold(BaseThreshold):
    """Manual Threshold."""

    def __init__(self, default_value: float, **kwargs) -> None:
        super().__init__(default_value, **kwargs)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Since this is manual threshold, updating it does not work."""
        return None

    def compute(self) -> Tensor:
        """Return the value of manual threshold."""
        return self.value
