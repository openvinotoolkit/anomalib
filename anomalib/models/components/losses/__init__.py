from typing import Optional

import torch


def dice(
    input_tensor: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    eps: float = 1e-8,
    reduction: Optional[str] = "mean",
) -> torch.Tensor:
    """
    Dice loss
    Args:
        input_tensor: annotated tensor
        target: mask tensor
        smooth: smoothing
        eps: epsilon
        reduction:

    Returns:
        The computed loss
    """
    bs = input_tensor.size(0)
    iflat = input_tensor.contiguous().view(bs, -1)
    tflat = target.contiguous().view(bs, -1)
    intersection = (iflat * tflat).sum(-1)
    loss = 1 - (2.0 * intersection + smooth) / (iflat.sum(-1) + tflat.sum(-1) + smooth + eps)

    if reduction == "mean":
        loss = loss.mean()
    return loss
