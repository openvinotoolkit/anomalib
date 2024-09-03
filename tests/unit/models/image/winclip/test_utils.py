"""Unit tests for WinCLIP utils."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from anomalib.models.image.winclip.utils import (
    class_scores,
    cosine_similarity,
    harmonic_aggregation,
    make_masks,
    visual_association_score,
)


class TestCosineSimilarity:
    """Unit tests for cosine similarity computation."""

    @staticmethod
    def test_computation() -> None:
        """Test cosine similarity computation."""
        input1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        input2 = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        assert torch.allclose(cosine_similarity(input1, input2), torch.tensor([[[0.0000, 0.7071], [1.0000, 0.7071]]]))

    @staticmethod
    def test_single_batch() -> None:
        """Test cosine similarity with single batch inputs."""
        input1 = torch.randn(1, 100, 128)
        input2 = torch.randn(1, 200, 128)
        assert cosine_similarity(input1, input2).shape == torch.Size([1, 100, 200])

    @staticmethod
    def test_multi_batch() -> None:
        """Test cosine similarity with multiple batch inputs."""
        input1 = torch.randn(10, 100, 128)
        input2 = torch.randn(10, 200, 128)
        assert cosine_similarity(input1, input2).shape == torch.Size([10, 100, 200])

    @staticmethod
    def test_2d() -> None:
        """Test cosine similarity with 2D input."""
        input1 = torch.randn(100, 128)
        input2 = torch.randn(200, 128)
        assert cosine_similarity(input1, input2).shape == torch.Size([100, 200])

    @staticmethod
    def test_2d_3d() -> None:
        """Test cosine similarity with 2D and 3D input."""
        input1 = torch.randn(100, 128)
        input2 = torch.randn(1, 200, 128)
        assert cosine_similarity(input1, input2).shape == torch.Size([100, 200])

    @staticmethod
    def test_3d_2d() -> None:
        """Test cosine similarity with 3D and 2D input."""
        input1 = torch.randn(10, 100, 128)
        input2 = torch.randn(200, 128)
        assert cosine_similarity(input1, input2).shape == torch.Size([10, 100, 200])


class TestClassScores:
    """Unit tests for CLIP class score computation."""

    @staticmethod
    def test_computation() -> None:
        """Test CLIP class score computation."""
        input1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        input2 = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        target = torch.tensor([[0.3302, 0.6698], [0.5727, 0.4273]])
        assert torch.allclose(class_scores(input1, input2), target, atol=1e-4)

    @staticmethod
    def test_called_with_target() -> None:
        """Test CLIP class score computation without target."""
        input1 = torch.randn(100, 128)
        input2 = torch.randn(200, 128)
        assert class_scores(input1, input2, target_class=0).shape == torch.Size([100])


class TestHarmonicAggregation:
    """Unit tests for harmonic aggregation computation."""

    @staticmethod
    def test_3x3_grid() -> None:
        """Test harmonic aggregation computation."""
        # example for a 3x3 patch grid with 4 sliding windows of size 2x2
        window_scores = torch.tensor([[1.0, 0.75, 0.5, 0.25]])
        output_size = (3, 3)
        masks = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7], [4, 5, 7, 8]])
        target = torch.tensor([[[1.0000, 0.8571, 0.7500], [0.6667, 0.4800, 0.3750], [0.5000, 0.3333, 0.2500]]])
        output = harmonic_aggregation(window_scores, output_size, masks)
        assert torch.allclose(output, target, atol=1e-4)

    @staticmethod
    def test_multi_batch() -> None:
        """Test harmonic aggregation computation with multiple batches."""
        window_scores = torch.randn(2, 4)
        output_size = (3, 3)
        masks = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7], [4, 5, 7, 8]])
        output = harmonic_aggregation(window_scores, output_size, masks)
        assert output.shape == torch.Size([2, 3, 3])


class TestVisualAssociationScore:
    """Unit tests for visual association score computation."""

    @staticmethod
    def test_computation() -> None:
        """Test visual association score computation."""
        embeddings = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        reference_embeddings = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]])
        target = torch.tensor([[0.1464, 0.0000]])
        assert torch.allclose(visual_association_score(embeddings, reference_embeddings), target, atol=1e-4)

    @staticmethod
    def test_multi_batch() -> None:
        """Test visual association score computation with multiple batches."""
        embeddings = torch.randn(10, 100, 128)
        reference_embeddings = torch.randn(2, 100, 128)
        assert visual_association_score(embeddings, reference_embeddings).shape == torch.Size([10, 100])


class TestMakeMasks:
    """Unit tests for mask generation."""

    @staticmethod
    def test_produces_correct_indices() -> None:
        """Test mask generation."""
        patch_grid_size = (3, 3)
        kernel_size = 2
        target = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7], [4, 5, 7, 8]])
        assert torch.equal(make_masks(patch_grid_size, kernel_size), target)

    @staticmethod
    @pytest.mark.parametrize(
        ("grid_size", "kernel_size", "stride", "target"),
        [
            ((2, 2), 2, 1, (4, 1)),
            ((3, 3), 2, 1, (4, 4)),
            ((4, 4), 2, 1, (4, 9)),
            ((3, 3), 1, 1, (1, 9)),
            ((4, 4), 2, 2, (4, 4)),
        ],
    )
    def test_shapes(grid_size: tuple[int, int], kernel_size: int, stride: int, target: tuple[int, int]) -> None:
        """Test mask generation for different grid sizes and kernel sizes."""
        assert make_masks(grid_size, kernel_size, stride).shape == target

    @staticmethod
    @pytest.mark.parametrize(
        ("grid_size", "kernel_size"),
        [
            ((2, 2), 3),
            ((2, 4), 3),
            ((4, 2), 3),
        ],
    )
    def test_raises_error_when_window_size_larger_than_grid_size(
        grid_size: tuple[int, int],
        kernel_size: int,
    ) -> None:
        """Test that an error is raised when the kernel size is larger than the grid size."""
        with pytest.raises(ValueError, match="Each dimension of the grid size must be greater than"):
            make_masks(grid_size, kernel_size)
