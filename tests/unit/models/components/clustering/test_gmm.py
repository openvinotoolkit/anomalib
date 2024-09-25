"""Unit tests for Anomalib's Gaussian Mixture Model."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import torch
from _pytest.logging import LogCaptureFixture

from anomalib.models.components.cluster.gmm import GaussianMixture


def test_fit_and_predict() -> None:
    """Test that the model can fit to some synthetic data and correctly predict the labels."""
    # Create a GaussianMixture model
    model = GaussianMixture(n_components=2)

    # Create some synthetic data
    data = torch.cat(
        [
            torch.randn(100, 2) + torch.tensor([10.0, 10.0]),
            torch.randn(100, 2) + torch.tensor([-10.0, -10.0]),
        ],
    )

    # Fit the model to the data
    model.fit(data)

    # Predict the labels of the data
    labels = model.predict(data)

    # Check that the labels are as expected
    assert (labels[:100] == labels[0]).all()
    assert (labels[100:] == labels[100]).all()
    assert (labels[:100] != labels[100:]).all()


def test_warns_on_non_convergence(caplog: LogCaptureFixture) -> None:
    """Test that the model warns when it does not converge."""
    # Create a GaussianMixture model
    model = GaussianMixture(n_components=2, n_iter=1)

    # Create some synthetic data
    data = torch.cat(
        [
            torch.randn(100, 2) + torch.tensor([2.0, 2.0]),
            torch.randn(100, 2) + torch.tensor([-2.0, -2.0]),
        ],
    )

    # Fit the model to the data
    with caplog.at_level(logging.WARNING):
        model.fit(data)

    assert "GMM did not converge" in caplog.text
