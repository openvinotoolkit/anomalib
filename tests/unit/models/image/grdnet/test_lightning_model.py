# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GRD-Net Lightning lifecycle."""

from copy import deepcopy
from typing import NamedTuple
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from anomalib import LearningType
from anomalib.data import ImageBatch, InferenceBatch
from anomalib.models.image.grdnet.lightning_model import GRDNet


class _TinyGenerator(nn.Module):
    """Small generator with batch normalization for optimizer-boundary tests."""

    def __init__(self) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(3, 3, kernel_size=1)
        self.normalization = nn.BatchNorm2d(3)

    def reconstruct(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a compact latent and bounded reconstruction."""
        reconstruction = torch.sigmoid(self.normalization(self.convolution(images)))
        return reconstruction.mean(dim=(-2, -1), keepdim=True), reconstruction

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return input latent, reconstruction, and reconstruction latent."""
        latent, reconstruction = self.reconstruct(images)
        reconstruction_latent = reconstruction.mean(dim=(-2, -1), keepdim=True)
        return latent, reconstruction, reconstruction_latent


class _TinyDiscriminator(nn.Module):
    """Small discriminator with observable batch-normalization state."""

    def __init__(self) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(3, 2, kernel_size=1)
        self.normalization = nn.BatchNorm2d(2)
        self.classifier = nn.Linear(2, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return spatial features and real/fake logits."""
        features = self.normalization(self.convolution(images))
        logits = self.classifier(features.mean(dim=(-2, -1)))
        return features, logits


class _TinySegmentator(nn.Module):
    """Small segmentator that records whether reconstruction input is detached."""

    def __init__(self) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(6, 2, kernel_size=1)
        self.reconstruction_requires_grad: bool | None = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return two-class logits and record the reconstruction gradient flag."""
        self.reconstruction_requires_grad = images[:, 3:].requires_grad
        return self.convolution(images)


class _TinyModel(nn.Module):
    """Small three-network GRD-Net replacement."""

    def __init__(self) -> None:
        super().__init__()
        self.generator = _TinyGenerator()
        self.discriminator = _TinyDiscriminator()
        self.segmentator = _TinySegmentator()

    def forward(self, images: torch.Tensor) -> InferenceBatch:  # noqa: PLR6301
        """Return deterministic standard anomaly predictions."""
        anomaly_map = images.mean(dim=1, keepdim=True)
        return InferenceBatch(
            pred_score=anomaly_map.amax(dim=(-2, -1)).squeeze(1),
            anomaly_map=anomaly_map,
        )


class _OptimizerConfig(NamedTuple):
    """Typed optimizer configuration used by tests."""

    optimizers: list[torch.optim.Optimizer]
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> GRDNet:
    """Create GRD-Net with small local subnetworks."""
    monkeypatch.setattr(GRDNet, "configure_model", staticmethod(_TinyModel))
    instance = GRDNet(pre_processor=False, post_processor=False, evaluator=False, visualizer=False)
    monkeypatch.setattr(instance, "manual_backward", lambda loss: loss.backward())

    def clip_generator_gradients(
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float,
        gradient_clip_algorithm: str,
    ) -> None:
        """Apply the production clipping operation without a Trainer."""
        del optimizer, gradient_clip_algorithm
        torch.nn.utils.clip_grad_norm_(instance.model.generator.parameters(), gradient_clip_val)

    monkeypatch.setattr(instance, "clip_gradients", clip_generator_gradients)
    return instance


@pytest.fixture
def optimizer_config(model: GRDNet) -> _OptimizerConfig:
    """Return the three optimizers and generator scheduler."""
    optimizers, scheduler_configs = model.configure_optimizers()
    scheduler = scheduler_configs[0]["scheduler"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    return _OptimizerConfig(optimizers, scheduler)


def _parameter_state(module: nn.Module) -> dict[str, torch.Tensor]:
    """Clone a module's trainable parameter state."""
    return {name: parameter.detach().clone() for name, parameter in module.named_parameters()}


def _state_changed(before: dict[str, torch.Tensor], module: nn.Module) -> bool:
    """Return whether any trainable parameter changed."""
    return any(not torch.equal(before[name], parameter) for name, parameter in module.named_parameters())


def test_properties(model: GRDNet) -> None:
    """GRD-Net exposes the expected learning and Trainer contracts."""
    assert model.learning_type is LearningType.ONE_CLASS
    assert model.trainer_arguments == {"gradient_clip_val": 0, "num_sanity_val_steps": 0}
    assert not model.automatic_optimization


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("texture_source", "unknown"),
        ("perlin_probability", -0.1),
        ("perlin_probability", 1.1),
        ("perlin_probability", True),
        ("adversarial_weight", -1.0),
        ("contextual_weight", float("inf")),
        ("encoder_weight", True),
        ("learning_rate", 0.0),
        ("learning_rate", float("nan")),
    ],
)
def test_constructor_validation(argument: str, value: object) -> None:
    """Invalid public numeric and texture options are rejected."""
    with pytest.raises(ValueError, match=r".+"):
        GRDNet(**{argument: value})


def test_optimizer_order_and_parameters(model: GRDNet, optimizer_config: _OptimizerConfig) -> None:
    """Optimizers are ordered by discriminator, generator, then segmentator."""
    expected_modules = (model.model.discriminator, model.model.generator, model.model.segmentator)
    for optimizer, module in zip(optimizer_config.optimizers, expected_modules, strict=True):
        optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        assert optimized == {id(parameter) for parameter in module.parameters()}
        assert optimizer.defaults["betas"] == (0.5, 0.999)
        assert optimizer.defaults["weight_decay"] == 0


def test_discriminator_phase_changes_only_discriminator(
    model: GRDNet,
    optimizer_config: _OptimizerConfig,
) -> None:
    """The first phase updates only discriminator parameters."""
    images = torch.rand(2, 3, 8, 8)
    before = {name: _parameter_state(module) for name, module in model.model.named_children()}

    model._update_discriminator(images, optimizer_config.optimizers[0])  # noqa: SLF001

    assert _state_changed(before["discriminator"], model.model.discriminator)
    assert not _state_changed(before["generator"], model.model.generator)
    assert not _state_changed(before["segmentator"], model.model.segmentator)


def test_generator_phase_freezes_discriminator(
    model: GRDNet,
    optimizer_config: _OptimizerConfig,
) -> None:
    """The second phase updates only the generator and freezes discriminator BN state."""
    images = torch.rand(2, 3, 8, 8)
    before = {name: _parameter_state(module) for name, module in model.model.named_children()}
    running_mean = model.model.discriminator.normalization.running_mean.clone()

    model._update_generator(images, optimizer_config.optimizers[1])  # noqa: SLF001

    assert _state_changed(before["generator"], model.model.generator)
    assert not _state_changed(before["discriminator"], model.model.discriminator)
    assert not _state_changed(before["segmentator"], model.model.segmentator)
    assert torch.equal(running_mean, model.model.discriminator.normalization.running_mean)
    assert all(parameter.requires_grad for parameter in model.model.discriminator.parameters())


def test_segmentator_phase_uses_detached_reconstruction(
    model: GRDNet,
    optimizer_config: _OptimizerConfig,
) -> None:
    """The third phase updates only the segmentator with a detached reconstruction."""
    images = torch.rand(2, 3, 8, 8)
    perturbed = torch.rand_like(images)
    masks = torch.ones(2, 1, 8, 8)
    before = {name: _parameter_state(module) for name, module in model.model.named_children()}

    model._update_segmentator(images, perturbed, masks, masks, optimizer_config.optimizers[2])  # noqa: SLF001

    assert _state_changed(before["segmentator"], model.model.segmentator)
    assert not _state_changed(before["generator"], model.model.generator)
    assert not _state_changed(before["discriminator"], model.model.discriminator)
    assert model.model.segmentator.reconstruction_requires_grad is False


def test_standard_batch_uses_full_roi() -> None:
    """Standard image batches receive an all-ones training ROI."""
    batch = ImageBatch(image=torch.rand(2, 3, 7, 9))
    roi_masks = GRDNet._roi_masks(batch)  # noqa: SLF001
    assert roi_masks.shape == (2, 1, 7, 9)
    assert roi_masks.dtype == batch.image.dtype
    assert roi_masks.all()


def test_generator_scheduler_uses_contextual_epoch_mean(
    model: GRDNet,
    optimizer_config: _OptimizerConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the generator learning rate follows the contextual-loss plateau."""
    discriminator_optimizer, generator_optimizer, segmentator_optimizer = optimizer_config.optimizers
    monkeypatch.setattr(model, "lr_schedulers", lambda: optimizer_config.scheduler)
    monkeypatch.setattr(model, "log", MagicMock())
    initial_rates = [optimizer.param_groups[0]["lr"] for optimizer in optimizer_config.optimizers]

    for _ in range(5):
        model.contextual_loss_mean.update(torch.tensor(1.0))
        model.on_train_epoch_end()

    assert discriminator_optimizer.param_groups[0]["lr"] == initial_rates[0]
    expected_generator_rate = initial_rates[1] * torch.exp(torch.tensor(-0.1)).item()
    assert generator_optimizer.param_groups[0]["lr"] == pytest.approx(expected_generator_rate)
    assert segmentator_optimizer.param_groups[0]["lr"] == initial_rates[2]
    assert model.contextual_loss_mean.update_count == 0


def test_validation_step_returns_standard_predictions(model: GRDNet) -> None:
    """Validation merges standard GRD-Net predictions into the input batch."""
    batch = ImageBatch(image=torch.rand(2, 3, 8, 10))
    result = model.validation_step(deepcopy(batch))
    assert result.pred_score.shape == (2,)
    assert result.anomaly_map.shape == (2, 8, 10)
