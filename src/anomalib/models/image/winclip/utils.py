"""WinCLIP utils."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn


def cosine_similarity(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix between two tensors.

    Computes the cosine similarity between all pairs of vectors in the two tensors.

    Args:
        input1 (torch.Tensor): Input tensor of shape (N, D) or (B, N, D).
        input2 (torch.Tensor): Input tensor of shape (M, D) or (B, M, D).

    Returns:
        torch.Tensor: Cosine similarity matrix of shape (N, M) or (B, N, M).

    Examples:
        >>> input1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> input2 = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        >>> cosine_similarity(input1, input2)
        tensor([[[0.0000, 0.7071],
                 [1.0000, 0.7071]]])

        >>> input1 = torch.randn(100, 128)
        >>> input2 = torch.randn(200, 128)
        >>> cosine_similarity(input1, input2).shape
        torch.Size([100, 200])

        >>> input1 = torch.randn(10, 100, 128)
        >>> input2 = torch.randn(10, 200, 128)
        >>> cosine_similarity(input1, input2).shape
        torch.Size([10, 100, 200])
    """
    ndim = input1.ndim
    input1 = input1.unsqueeze(0) if input1.ndim == 2 else input1
    input2 = input2.repeat(input1.shape[0], 1, 1) if input2.ndim == 2 else input2

    input1_norm = nn.functional.normalize(input1, p=2, dim=-1)
    input2_norm = nn.functional.normalize(input2, p=2, dim=-1)
    similarity = torch.bmm(input1_norm, input2_norm.transpose(-2, -1))
    if ndim == 2:
        return similarity.squeeze(0)
    return similarity


def class_scores(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float = 1.0,
    target_class: int | None = None,
) -> torch.Tensor:
    """Compute class scores between a set of N image embeddings and a set of M text embeddings.

    Each text embedding represents the embedding of a prompt for a specific class. By computing the cosine similarity
    between each image embedding and each text embedding, we obtain a similarity matrix of shape (N, M). This matrix is
    then used to compute the confidence scores for each class by scaling by a temperature parameter and applying the
    softmax function (Equation (1) in the WinCLIP paper).

    Args:
        image_embeddings (torch.Tensor): Image embedding matrix of shape (N, D) or (B, N, D).
        text_embeddings (torch.Tensor): Text embedding matrix of shape (M, D) or (B, M, D).
        temperature (float): Temperature hyperparameter.
        target_class (int): Index of the target class. If None, the scores for all classes are returned.

    Returns:
        torch.Tensor: Similarity score of shape (N, M) or (B, N, M).

    Examples:
        >>> image_embeddings = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> text_embeddings = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        >>> class_scores(image_embeddings, text_embeddings)
        tensor([[0.3302, 0.6698],
                [0.5727, 0.4273]])

        >>> image_embeddings = torch.randn(100, 128)
        >>> text_embeddings = torch.randn(200, 128)
        >>> class_scores(image_embeddings, text_embeddings).shape
        torch.Size([100, 200])

        >>> image_embeddings = torch.randn(10, 100, 128)
        >>> text_embeddings = torch.randn(10, 200, 128)
        >>> class_scores(image_embeddings, text_embeddings).shape
        torch.Size([10, 100, 200])

        >>> image_embeddings = torch.randn(10, 100, 128)
        >>> text_embeddings = torch.randn(10, 200, 128)
        >>> class_scores(image_embeddings, text_embeddings, target_class=0).shape
        torch.Size([10, 100])
    """
    scores = (cosine_similarity(image_embeddings, text_embeddings) / temperature).softmax(dim=-1)
    if target_class is not None:
        return scores[..., target_class]
    return scores


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


def make_masks(grid_size: tuple[int, int], kernel_size: int, stride: int = 1) -> torch.Tensor:
    """Make a set of mask to select patches from a feature map in a sliding window fashion.

    Args:
        grid_size (tuple[int, int]): The shape of the feature map.
        kernel_size (int): The size of the kernel in number of patches.
        stride (int): The size of the stride in number of patches.

    Returns:
        torch.Tensor: Set of masks of shape (1, n_patches_per_mask, n_masks).
    """
    height, width = grid_size
    grid = torch.arange(1, height * width + 1).reshape(1, 1, height, width)
    masks = nn.functional.unfold(grid.float(), kernel_size=kernel_size, stride=stride).int()
    return masks.squeeze()
