"""Unit tests for Anomalib's Gaussian Mixture Model."""

import torch

from anomalib.models.components.cluster.gmm import GaussianMixture


def test_fit_and_predict() -> None:
    """Test that the model can fit to some synthetic data and correctly predict the labels."""
    # Create a GaussianMixture model
    model = GaussianMixture(n_components=2)

    # Create some synthetic data
    data = torch.cat(
        [
            torch.randn(100, 2) + torch.tensor([2.0, 2.0]),
            torch.randn(100, 2) + torch.tensor([-2.0, -2.0]),
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
