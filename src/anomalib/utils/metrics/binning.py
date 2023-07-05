from torch import Tensor, linspace, all


def thresholds_between_min_and_max(preds: Tensor, num_thresholds: int = 100) -> Tensor:
    return linspace(
        start=preds.min(),
        end=preds.max(),
        steps=num_thresholds,
    )


def thresholds_between_0_and_1(num_thresholds: int = 100) -> Tensor:
    return linspace(
        start=0,
        end=1,
        steps=num_thresholds,
    )