"""WinCLIP utils."""

import torch
import torch.nn.functional as F


def cosine_similarity(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity matrix between two tensors.

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
    return torch.bmm(input1_norm, input2_norm.transpose(-2, -1))


def simmilarity_score(input1: torch.Tensor, input2: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    """Compute similarity score between two tensors.

    The similarity score is computed as the softmax of the cosine similarity matrix, scaled by a temperature parameter.
    
    Args:
        input1 (torch.Tensor): Input tensor of shape (N, D) or (B, N, D).
        input2 (torch.Tensor): Input tensor of shape (M, D) or (B, M, D).
        temp (float): Temperature hyperparameter.
    Returns:
        torch.Tensor: Similarity score of shape (N, M).
    """
    return (cosine_similarity(input1, input2) / temp).softmax(dim=-1).squeeze()


def harmonic_aggregation(window_scores: torch.Tensor, output_size: tuple, masks: torch.Tensor) -> torch.Tensor:
    """Perform harmonic aggregation on window scores.

    Computes a single score for each patch location by aggregating the scores of all windows that cover the patch.
    Scores are aggregated using the harmonic mean.

    Args:
        window_scores (torch.Tensor): Tensor of shape (batch_size, n_masks) representing the scores for each sliding 
            window location.
        output_size (tuple): Tuple of integers representing the output size (H, W).
        masks (torch.Tensor): Tensor of shape (n_patches_per_mask, n_masks) representing the masks. Each mask is
            set of indices indicating which patches are covered by the mask.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, H, W) representing the aggregated scores.
    """
    batch_size = window_scores.shape[0]
    height, width = output_size
    
    scores = []
    for idx in range(height * width):
        patch_mask = torch.any(masks == idx + 1, dim=0)  # boolean tensor indicating which masks contain the patch
        scores.append(sum(patch_mask) / (1 / window_scores.T[patch_mask]).sum(dim=0))

    return torch.stack(scores).T.reshape(batch_size, height, width).nan_to_num(posinf=0.0)


def visual_association_score(embeddings: torch.Tensor, reference_embeddings: torch.Tensor) -> torch.Tensor:
    """Compute visual association scores between a set of embeddings and a set of reference embeddings.
    
    Args:
        embeddings (torch.Tensor): Tensor of shape (batch_size, n_embeddings, dimensionality) representing the
            embeddings.
        reference_embeddings (torch.Tensor): Tensor of shape (n_patches, n_embeddings) representing the reference 
            embeddings.
    
    Returns:
        torch.Tensor: Tensor of shape (batch_size, n_embeddings) representing the visual association scores.
    """
    reference_embeddings = reference_embeddings.reshape(-1, embeddings.shape[-1])
    scores = cosine_similarity(embeddings, reference_embeddings)
    return (1 - scores).min(dim=-1)[0] / 2


def make_masks(grid_size: tuple, kernel_size: int, stride: int = 1) -> torch.Tensor:
    """Make a set of mask to select patches from a feature map in a sliding window fashion.

    Args:
        grid_size tuple(int, int): The shape of the feature map.
        kernel_size (int): The size of the kernel in number of patches.
        stride (int): The size of the stride in number of patches.

    Returns:
        torch.Tensor: Set of masks of shape (1, n_patches_per_mask, n_masks).
    """
    height, width = grid_size
    grid = torch.arange(1, height * width + 1).reshape(1, 1, height, width)
    masks = F.unfold(grid.float(), kernel_size=kernel_size, stride=stride).int()
    return masks.squeeze()
