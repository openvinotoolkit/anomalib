from torch import Tensor, linspace
from torch import device as torch_device


def thresholds_between_min_and_max(
    preds: Tensor,
    num_thresholds: int = 100,
    device: torch_device | None = None,
) -> Tensor:
    return linspace(start=preds.min(), end=preds.max(), steps=num_thresholds, device=device)


def thresholds_between_0_and_1(num_thresholds: int = 100, device: torch_device | None = None) -> Tensor:
    return linspace(start=0, end=1, steps=num_thresholds, device=device)
