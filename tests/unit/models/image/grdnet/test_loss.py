# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GRD-Net training objectives."""

import torch
from kornia.losses import FocalLoss, SSIMLoss
from torch.nn import functional

from anomalib.models.image.grdnet.loss import (
    GRDNetDiscriminatorLoss,
    GRDNetGeneratorLoss,
    GRDNetSegmentatorLoss,
)


def test_generator_loss_components_and_weights() -> None:
    """The generator objective should expose the three paper-selected weighted terms."""
    criterion = GRDNetGeneratorLoss(adversarial_weight=2.0, contextual_weight=3.0, encoder_weight=4.0)
    image = torch.zeros((1, 1, 11, 11))
    reconstruction = torch.ones_like(image)
    latent = torch.zeros((1, 1, 2, 2))
    reconstruction_latent = torch.full_like(latent, 3.0)
    real_features = torch.zeros((1, 2))
    fake_features = torch.full_like(real_features, 2.0)

    losses = criterion(image, reconstruction, latent, reconstruction_latent, real_features, fake_features)

    expected_contextual = torch.tensor(1.0) + 2 * SSIMLoss(window_size=11)(reconstruction, image)
    assert torch.isclose(losses.adversarial, torch.tensor(4.0))
    assert torch.isclose(losses.contextual, expected_contextual)
    assert torch.isclose(losses.encoder, torch.tensor(3.0))
    assert torch.isclose(losses.total, 2 * losses.adversarial + 3 * losses.contextual + 4 * losses.encoder)


def test_generator_loss_stops_real_feature_gradient() -> None:
    """Feature matching should backpropagate only through fake discriminator features."""
    criterion = GRDNetGeneratorLoss()
    real_features = torch.tensor([[1.0]], requires_grad=True)
    fake_features = torch.tensor([[0.0]], requires_grad=True)
    zeros = torch.zeros((1, 1, 11, 11), requires_grad=True)

    losses = criterion(zeros, zeros, zeros, zeros, real_features, fake_features)
    losses.total.backward()

    assert real_features.grad is None
    assert fake_features.grad is not None
    assert torch.isfinite(fake_features.grad).all()


def test_discriminator_loss_matches_binary_cross_entropy() -> None:
    """The discriminator objective should average real and fake logit losses."""
    real_logits = torch.tensor([[0.0], [2.0]])
    fake_logits = torch.tensor([[0.0], [-2.0]])

    actual = GRDNetDiscriminatorLoss()(real_logits, fake_logits)
    expected = 0.5 * (functional.softplus(-real_logits).mean() + functional.softplus(fake_logits).mean())

    assert torch.isclose(actual, expected)


def test_segmentator_target_intersects_anomaly_and_roi() -> None:
    """ROI supervision should modify the target rather than the prediction."""
    anomaly_mask = torch.tensor([[[[1.0, 1.0], [0.0, 1.0]]]])
    roi_mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

    target = GRDNetSegmentatorLoss.target(anomaly_mask, roi_mask)

    assert torch.equal(target, torch.tensor([[[1, 0], [0, 0]]]))


def test_full_roi_matches_unmasked_target() -> None:
    """A full-image ROI should preserve the synthetic anomaly target."""
    anomaly_mask = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

    target = GRDNetSegmentatorLoss.target(anomaly_mask, torch.ones_like(anomaly_mask))

    assert torch.equal(target, anomaly_mask.squeeze(1).long())


def test_empty_roi_produces_finite_all_normal_loss() -> None:
    """An empty ROI should yield an all-normal target and a finite focal loss."""
    criterion = GRDNetSegmentatorLoss()
    anomaly_mask = torch.ones((1, 1, 2, 2))
    roi_mask = torch.zeros_like(anomaly_mask)
    logits = torch.tensor([[[[0.1, -0.2], [0.3, -0.4]], [[-0.1, 0.2], [-0.3, 0.4]]]])

    target = criterion.target(anomaly_mask, roi_mask)
    loss = criterion(logits, anomaly_mask, roi_mask)

    assert torch.count_nonzero(target) == 0
    assert torch.isfinite(loss)


def test_segmentator_uses_target_side_roi_masking() -> None:
    """Target-side masking should differ from masking the prediction logits."""
    criterion = GRDNetSegmentatorLoss()
    anomaly_mask = torch.ones((1, 1, 2, 2))
    roi_mask = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    logits = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]], [[0.9, 0.8], [0.7, 0.6]]]])

    actual = criterion(logits, anomaly_mask, roi_mask)
    prediction_masked = FocalLoss(alpha=1.0, gamma=2.0, reduction="mean")(
        logits * roi_mask,
        anomaly_mask.squeeze(1).long(),
    )

    assert not torch.isclose(actual, prediction_masked)
