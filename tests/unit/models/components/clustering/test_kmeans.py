# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for KMeans clustering."""

from unittest.mock import patch

import torch

from anomalib.models.components.cluster import KMeans


def test_kmeans_fit_shapes_and_dtypes() -> None:
    """Test that fit returns tensors with expected shapes and dtypes."""
    n_clusters = 3
    n_samples = 100
    n_features = 5

    kmeans = KMeans(n_clusters=n_clusters)
    data = torch.randn(n_samples, n_features)

    labels, centers = kmeans.fit(data)

    assert isinstance(labels, torch.Tensor)
    assert isinstance(centers, torch.Tensor)

    assert labels.shape == (n_samples,)
    assert labels.dtype == torch.int64

    assert centers.shape == (n_clusters, n_features)
    assert centers.dtype == data.dtype


def test_kmeans_predict_consistency() -> None:
    """Test that predict is consistent with fitted centers on synthetic data."""
    n_clusters = 2

    kmeans = KMeans(n_clusters=n_clusters)

    # Create distinct clusters
    data = torch.cat(
        [
            torch.randn(100, 2) * 0.1 + torch.tensor([5.0, 5.0]),
            torch.randn(100, 2) * 0.1 + torch.tensor([-5.0, -5.0]),
        ],
    )

    labels_fit, centers = kmeans.fit(data)

    # Predict on the same data
    labels_predict = kmeans.predict(data)

    # Predict labels should match fit labels
    assert torch.equal(labels_fit, labels_predict)

    # Test on new distinct points
    new_data = torch.tensor(
        [
            [5.1, 4.9],  # Close to cluster 0 (or 1)
            [-4.9, -5.1],  # Close to other cluster
        ],
    )

    new_predictions = kmeans.predict(new_data)

    # Find which center is closer to [5.0, 5.0]
    dist_to_center_0 = torch.norm(centers[0] - torch.tensor([5.0, 5.0]))
    dist_to_center_1 = torch.norm(centers[1] - torch.tensor([5.0, 5.0]))

    expected_label_first_point = 0 if dist_to_center_0 < dist_to_center_1 else 1
    expected_label_second_point = 1 - expected_label_first_point

    assert new_predictions[0].item() == expected_label_first_point
    assert new_predictions[1].item() == expected_label_second_point


def test_kmeans_early_exit() -> None:
    """Test that KMeans exits early if labels stop changing."""
    n_clusters = 2
    max_iter = 10

    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter)

    # Create easily separable data that will converge quickly
    data = torch.cat(
        [
            torch.randn(10, 2) * 0.01 + torch.tensor([5.0, 5.0]),
            torch.randn(10, 2) * 0.01 + torch.tensor([-5.0, -5.0]),
        ],
    )

    with patch("torch.cdist", wraps=torch.cdist) as mock_cdist:
        kmeans.fit(data)

        # It should exit early, so the number of iterations (and thus cdist calls)
        # should be less than max_iter + 1 (the final distance calculation)
        assert mock_cdist.call_count < max_iter + 1
