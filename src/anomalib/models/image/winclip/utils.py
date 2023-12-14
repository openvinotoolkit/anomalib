"""WinCLIP utils."""

import torch
import torch.nn.functional as F


def cosine_similarity(input1, input2):
    """Compute cosine similarity matrix between two tensors.

    # TODO: add examples

    Args:
        input1 (torch.Tensor): Input tensor of shape (N, D) or (B, N, D).
        input2 (torch.Tensor): Input tensor of shape (M, D) or (B, M, D).

    Returns:
        torch.Tensor: Cosine similarity matrix of shape (N, M).
    """
    input1 = input1.unsqueeze(0) if input1.ndim == 2 else input1
    input2 = input2.repeat(input1.shape[0], 1, 1) if input2.ndim == 2 else input2

    input1_norm = F.normalize(input1, p=2, dim=-1)
    input2_norm = F.normalize(input2, p=2, dim=-1)
    # TODO check if we can remove 0.07 factor and softmax
    return (torch.bmm(input1_norm, input2_norm.transpose(-2, -1)) / 0.07).softmax(dim=-1).squeeze()
