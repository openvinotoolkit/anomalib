from typing import Optional

from torch import Tensor, linspace
from torch import device as torch_device


def thresholds_between_min_and_max(
    preds: Tensor, num_thresholds: int = 100, device: Optional[torch_device] = None
) -> Tensor:
    return linspace(start=preds.min(), end=preds.max(), steps=num_thresholds, device=device)


def thresholds_between_0_and_1(num_thresholds: int = 100, device: Optional[torch_device] = None) -> Tensor:
    return linspace(start=0, end=1, steps=num_thresholds, device=device)
