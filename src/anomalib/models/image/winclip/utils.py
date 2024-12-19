"""Utility functions for WinCLIP model.

This module provides utility functions used by the WinCLIP model for anomaly detection:

- :func:`cosine_similarity`: Compute pairwise cosine similarity between tensors
- :func:`class_scores`: Calculate anomaly scores from CLIP embeddings
- :func:`harmonic_aggregation`: Aggregate scores using harmonic mean
- :func:`make_masks`: Generate sliding window masks
- :func:`visual_association_score`: Compute visual association scores

Example:
    >>> import torch
    >>> from anomalib.models.image.winclip.utils import cosine_similarity
    >>> input1 = torch.randn(100, 128)  # doctest: +SKIP
    >>> input2 = torch.randn(200, 128)  # doctest: +SKIP
    >>> similarity = cosine_similarity(input1, input2)  # doctest: +SKIP

See Also:
    - :class:`WinClip`: Main model class using these utilities
    - :class:`WinClipModel`: PyTorch model implementation
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn


def cosine_similarity(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix between two tensors.

    Computes the cosine similarity between all pairs of vectors in the two input tensors.
    The inputs can be either 2D or 3D tensors. For 2D inputs, an implicit batch
    dimension of 1 is added.

    Args:
        input1 (torch.Tensor): First input tensor of shape ``(N, D)`` or ``(B, N, D)``,
            where:
            - ``B`` is the optional batch dimension
            - ``N`` is the number of vectors in first input
            - ``D`` is the dimension of each vector
        input2 (torch.Tensor): Second input tensor of shape ``(M, D)`` or ``(B, M, D)``,
            where:
            - ``B`` is the optional batch dimension
            - ``M`` is the number of vectors in second input
            - ``D`` is the dimension of each vector (must match input1)

    Returns:
        torch.Tensor: Cosine similarity matrix of shape ``(N, M)`` for 2D inputs or
            ``(B, N, M)`` for 3D inputs, where each element ``[i,j]`` is the cosine
            similarity between vector ``i`` from ``input1`` and vector ``j`` from
            ``input2``.

    Examples:
        2D inputs (single batch):

        >>> input1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> input2 = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        >>> cosine_similarity(input1, input2)
        tensor([[[0.0000, 0.7071],
                 [1.0000, 0.7071]]])

        Different sized inputs:

        >>> input1 = torch.randn(100, 128)  # 100 vectors of dimension 128
        >>> input2 = torch.randn(200, 128)  # 200 vectors of dimension 128
        >>> similarity = cosine_similarity(input1, input2)
        >>> similarity.shape
        torch.Size([100, 200])

        3D inputs (batched):

        >>> input1 = torch.randn(10, 100, 128)  # 10 batches of 100 vectors
        >>> input2 = torch.randn(10, 200, 128)  # 10 batches of 200 vectors
        >>> similarity = cosine_similarity(input1, input2)
        >>> similarity.shape
        torch.Size([10, 100, 200])

    Note:
        The function automatically handles both 2D and 3D inputs by adding a batch
        dimension to 2D inputs. The vector dimension ``D`` must match between inputs.
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
    """Compute class scores between image embeddings and text embeddings.

    Computes similarity scores between image and text embeddings by first calculating
    cosine similarity and then applying temperature scaling and softmax. This follows
    Equation (1) in the WinCLIP paper.

    Each text embedding represents a prompt for a specific class. The similarity matrix
    is used to compute confidence scores for each class.

    Args:
        image_embeddings (torch.Tensor): Image embeddings with shape ``(N, D)`` or
            ``(B, N, D)``, where:
            - ``B`` is optional batch dimension
            - ``N`` is number of image embeddings
            - ``D`` is embedding dimension
        text_embeddings (torch.Tensor): Text embeddings with shape ``(M, D)`` or
            ``(B, M, D)``, where:
            - ``B`` is optional batch dimension
            - ``M`` is number of text embeddings
            - ``D`` is embedding dimension (must match image embeddings)
        temperature (float, optional): Temperature scaling parameter. Higher values
            make distribution more uniform, lower values make it more peaked.
            Defaults to ``1.0``.
        target_class (int | None, optional): Index of target class. If provided,
            returns scores only for that class. Defaults to ``None``.

    Returns:
        torch.Tensor: Class similarity scores. Shape depends on inputs and
            ``target_class``:
            - If no target class: ``(N, M)`` or ``(B, N, M)``
            - If target class specified: ``(N,)`` or ``(B, N)``

    Examples:
        Basic usage with 2D inputs:

        >>> image_embeddings = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        >>> text_embeddings = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        >>> class_scores(image_embeddings, text_embeddings)
        tensor([[0.3302, 0.6698],
                [0.5727, 0.4273]])

        With different sized inputs:

        >>> image_embeddings = torch.randn(100, 128)  # 100 vectors
        >>> text_embeddings = torch.randn(200, 128)  # 200 class prompts
        >>> class_scores(image_embeddings, text_embeddings).shape
        torch.Size([100, 200])

        With batched 3D inputs:

        >>> image_embeddings = torch.randn(10, 100, 128)  # 10 batches
        >>> text_embeddings = torch.randn(10, 200, 128)  # 10 batches
        >>> class_scores(image_embeddings, text_embeddings).shape
        torch.Size([10, 100, 200])

        With target class specified:

        >>> scores = class_scores(image_embeddings, text_embeddings, target_class=0)
        >>> scores.shape
        torch.Size([10, 100])
    """
    scores = (cosine_similarity(image_embeddings, text_embeddings) / temperature).softmax(dim=-1)
    if target_class is not None:
        return scores[..., target_class]
    return scores


def harmonic_aggregation(window_scores: torch.Tensor, output_size: tuple, masks: torch.Tensor) -> torch.Tensor:
    """Perform harmonic aggregation on window scores.

    Computes a single score for each patch location by aggregating the scores of all
    windows that cover the patch. Scores are aggregated using the harmonic mean.

    Args:
        window_scores (torch.Tensor): Scores for each sliding window location.
            Shape: ``(batch_size, n_masks)``.
        output_size (tuple): Output dimensions ``(H, W)``.
        masks (torch.Tensor): Binary masks indicating which patches are covered by each
            window. Shape: ``(n_patches_per_mask, n_masks)``.

    Returns:
        torch.Tensor: Aggregated scores. Shape: ``(batch_size, H, W)``.

    Example:
        Example for a 3x3 patch grid with 4 sliding windows of size 2x2:

        >>> window_scores = torch.tensor([[1.0, 0.75, 0.5, 0.25]])
        >>> output_size = (3, 3)
        >>> masks = torch.Tensor([[0, 1, 3, 4],
        ...                      [1, 2, 4, 5],
        ...                      [3, 4, 6, 7],
        ...                      [4, 5, 7, 8]])
        >>> harmonic_aggregation(window_scores, output_size, masks)
        tensor([[[1.0000, 0.8571, 0.7500],
                 [0.6667, 0.4800, 0.3750],
                 [0.5000, 0.3333, 0.2500]]])

    Note:
        The harmonic mean is used instead of arithmetic mean as it is more sensitive to
        low scores, making it better suited for anomaly detection where we want to
        emphasize potential defects.
    """
    batch_size = window_scores.shape[0]
    height, width = output_size

    scores = []
    for idx in range(height * width):
        patch_mask = torch.any(masks == idx, dim=0)  # boolean tensor indicating which masks contain the patch
        scores.append(sum(patch_mask) / (1 / window_scores.T[patch_mask]).sum(dim=0))

    return torch.stack(scores).T.reshape(batch_size, height, width).nan_to_num(posinf=0.0)


def visual_association_score(embeddings: torch.Tensor, reference_embeddings: torch.Tensor) -> torch.Tensor:
    """Compute visual association scores between a set of embeddings and a set of reference embeddings.

    Returns a visual association score for each patch location in the inputs. The visual association score is the
    minimum cosine distance between each embedding and the reference embeddings. Equation (4) in the paper.

    Args:
        embeddings (torch.Tensor): Tensor of shape ``(batch_size, n_patches, dimensionality)`` representing the
            embeddings.
        reference_embeddings (torch.Tensor): Tensor of shape ``(n_reference_embeddings, n_patches, dimensionality)``
            representing the reference embeddings.

    Returns:
        torch.Tensor: Tensor of shape ``(batch_size, n_patches)`` representing the visual association scores.

    Examples:
        >>> embeddings = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        >>> reference_embeddings = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]])
        >>> visual_association_score(embeddings, reference_embeddings)
        tensor([[0.1464, 0.0000]])

        >>> embeddings = torch.randn(10, 100, 128)
        >>> reference_embeddings = torch.randn(2, 100, 128)
        >>> visual_association_score(embeddings, reference_embeddings).shape
        torch.Size([10, 100])
    """
    reference_embeddings = reference_embeddings.reshape(-1, embeddings.shape[-1])
    scores = cosine_similarity(embeddings, reference_embeddings)
    return (1 - scores).min(dim=-1)[0] / 2


def make_masks(grid_size: tuple[int, int], kernel_size: int, stride: int = 1) -> torch.Tensor:
    """Make masks to select patches from a feature map using sliding windows.

    Creates a set of masks for selecting patches from a feature map in a sliding window
    fashion. Each column in the returned tensor represents one mask. A mask consists of
    indices indicating which patches are covered by that sliding window position.

    The number of masks equals the number of possible sliding window positions that fit
    in the feature map given the kernel size and stride.

    Args:
        grid_size (tuple[int, int]): Height and width of the feature map grid as
            ``(H, W)``.
        kernel_size (int): Size of the sliding window kernel in number of patches.
        stride (int, optional): Stride of the sliding window in number of patches.
            Defaults to ``1``.

    Returns:
        torch.Tensor: Set of masks with shape ``(n_patches_per_mask, n_masks)``. Each
            column represents indices of patches covered by one sliding window position.

    Raises:
        ValueError: If any dimension of ``grid_size`` is smaller than ``kernel_size``.

    Examples:
        Create masks for a 3x3 grid with kernel size 2 and stride 1:

        >>> make_masks((3, 3), 2)
        tensor([[0, 1, 3, 4],
                [1, 2, 4, 5],
                [3, 4, 6, 7],
                [4, 5, 7, 8]], dtype=torch.int32)

        Create masks for a 4x4 grid with kernel size 2 and stride 1:

        >>> make_masks((4, 4), 2)
        tensor([[ 0,  1,  2,  4,  5,  6,  8,  9, 10],
                [ 1,  2,  3,  5,  6,  7,  9, 10, 11],
                [ 4,  5,  6,  8,  9, 10, 12, 13, 14],
                [ 5,  6,  7,  9, 10, 11, 13, 14, 15]], dtype=torch.int32)

        Create masks for a 4x4 grid with kernel size 2 and stride 2:

        >>> make_masks((4, 4), 2, stride=2)
        tensor([[ 0,  2,  8, 10],
                [ 1,  3,  9, 11],
                [ 4,  6, 12, 14],
                [ 5,  7, 13, 15]], dtype=torch.int32)

    Note:
        The returned masks can be used with :func:`visual_association_score` to compute
        scores for sliding window positions.
    """
    if any(dim < kernel_size for dim in grid_size):
        msg = (
            "Each dimension of the grid size must be greater than or equal to "
            f"the kernel size. Got grid size {grid_size} and kernel size {kernel_size}."
        )
        raise ValueError(msg)
    height, width = grid_size
    grid = torch.arange(height * width).reshape(1, height, width)
    return nn.functional.unfold(grid.float(), kernel_size=kernel_size, stride=stride).int()
