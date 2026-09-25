# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GRD-Net Lightning lifecycle."""

from copy import deepcopy
from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, Normalize, Resize

from anomalib import LearningType
from anomalib.data import ImageBatch, ImageItem, InferenceBatch
from anomalib.data.dataclasses.torch.grdnet import GRDNetBatch, GRDNetItem
from anomalib.engine import Engine
from anomalib.models.image.grdnet import lightning_model
from anomalib.models.image.grdnet.lightning_model import GRDNet
from anomalib.models.image.grdnet.loss import GeneratorLosses
from anomalib.pre_processing import PreProcessor


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
        self.tile_size = (128, 128)
        self.stride = (64, 64)
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


class _TinyAnomalyGenerator(nn.Module):
    """Deterministic anomaly generator for lifecycle tests."""

    def forward(  # noqa: PLR6301
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Corrupt a fixed central region of every tile."""
        textures = torch.zeros_like(images)
        masks = torch.zeros(
            images.shape[0],
            1,
            *images.shape[-2:],
            device=images.device,
            dtype=images.dtype,
        )
        masks[..., 32:96, 32:96] = 1
        beta = torch.full(
            (images.shape[0], 1, 1, 1),
            0.75,
            device=images.device,
            dtype=images.dtype,
        )
        perturbed = images * (1 - masks) + images * masks * (1 - beta)
        return perturbed, textures, masks, beta


class _OptimizerConfig(NamedTuple):
    """Typed optimizer configuration used by tests."""

    optimizers: list[torch.optim.Optimizer]
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> GRDNet:
    """Create GRD-Net with small local subnetworks."""
    monkeypatch.setattr(GRDNet, "configure_torch_model", staticmethod(_TinyModel))

    def configure_anomaly_generator(texture_source: str, probability: float) -> _TinyAnomalyGenerator:
        """Create the deterministic test anomaly generator."""
        del texture_source, probability
        return _TinyAnomalyGenerator()

    monkeypatch.setattr(
        GRDNet,
        "configure_anomaly_generator",
        staticmethod(configure_anomaly_generator),
    )
    instance = GRDNet(pre_processor=False, post_processor=False, evaluator=False, visualizer=False)
    monkeypatch.setattr(instance, "manual_backward", lambda loss: loss.backward())
    return instance


@pytest.fixture
def optimizer_config(model: GRDNet) -> _OptimizerConfig:
    """Return the three optimizers and generator scheduler."""
    optimizers, schedulers = model.configure_optimizers()
    return _OptimizerConfig(optimizers, schedulers[0])


def _parameter_state(module: nn.Module) -> dict[str, torch.Tensor]:
    """Clone a module's trainable parameter state."""
    return {name: parameter.detach().clone() for name, parameter in module.named_parameters()}


def _state_changed(before: dict[str, torch.Tensor], module: nn.Module) -> bool:
    """Return whether any trainable parameter changed."""
    return any(not torch.equal(before[name], parameter) for name, parameter in module.named_parameters())


def _assert_nested_equal(actual: object, expected: object) -> None:
    """Assert equality for nested checkpoint state containing tensors."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


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


def test_training_uses_low_level_model_geometry(
    model: GRDNet,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training tiles images using the geometry selected by the Torch model."""
    model.model.tile_size = (256, 256)
    model.model.stride = (256, 256)
    captured: dict[str, object] = {}
    tile_image_and_roi = lightning_model.tile_image_and_roi

    def capture_geometry(
        images: torch.Tensor,
        roi_masks: torch.Tensor,
        tile_size: tuple[int, int],
        stride: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        captured.update({"tile_size": tile_size, "stride": stride})
        image_tiles, roi_tiles = tile_image_and_roi(images, roi_masks, tile_size=tile_size, stride=stride)
        captured["tile_count"] = image_tiles.shape[0]
        return image_tiles, roi_tiles

    loss = torch.tensor(1.0)
    generator_losses = GeneratorLosses(loss, loss, loss, loss)
    monkeypatch.setattr(lightning_model, "tile_image_and_roi", capture_geometry)
    monkeypatch.setattr(lightning_model, "rotate_image_and_roi", lambda images, roi_masks: (images, roi_masks))
    monkeypatch.setattr(model, "optimizers", lambda: (MagicMock(), MagicMock(), MagicMock()))
    monkeypatch.setattr(model, "_update_discriminator", MagicMock(return_value=loss))
    monkeypatch.setattr(model, "_update_generator", MagicMock(return_value=generator_losses))
    monkeypatch.setattr(model, "_update_segmentator", MagicMock(return_value=loss))
    monkeypatch.setattr(model, "log_dict", MagicMock())

    model.training_step(ImageBatch(image=torch.rand(1, 3, 256, 256)), 0)

    assert captured == {"tile_size": (256, 256), "stride": (256, 256), "tile_count": 1}


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


def _loader(*, with_roi: bool) -> DataLoader:
    """Create one deterministic image or ROI-aware batch."""
    image = torch.linspace(0, 1, 3 * 256 * 256).reshape(3, 256, 256)
    if with_roi:
        item = GRDNetItem(image=image, roi_mask=torch.ones(256, 256))
        return DataLoader([item], batch_size=1, collate_fn=GRDNetBatch.collate)
    return DataLoader([ImageItem(image=image)], batch_size=1, collate_fn=ImageBatch.collate)


def _engine(root: Path, *, max_epochs: int = 1) -> Engine:
    """Create a minimal CPU Engine for lifecycle tests."""
    return Engine(
        accelerator="cpu",
        devices=1,
        logger=False,
        default_root_dir=root,
        max_epochs=max_epochs,
        limit_train_batches=1,
        limit_val_batches=1,
        enable_model_summary=False,
    )


@pytest.mark.parametrize("with_roi", [False, True])
def test_engine_fit_with_standard_and_roi_batches(
    model: GRDNet,
    tmp_path: Path,
    with_roi: bool,
) -> None:
    """Engine fits and validates one ordinary or ROI-aware batch."""
    loader = _loader(with_roi=with_roi)
    engine = _engine(tmp_path / str(with_roi))
    engine.fit(model=model, train_dataloaders=loader, val_dataloaders=loader)
    assert engine.trainer.global_step == 3


def test_checkpoint_roundtrip_restores_training_state(
    model: GRDNet,
    tmp_path: Path,
) -> None:
    """Engine checkpoints restore networks, optimizers, scheduler, and predictions."""
    loader = _loader(with_roi=False)
    engine = _engine(tmp_path / "fit")
    engine.fit(model=model, train_dataloaders=loader, val_dataloaders=loader)
    predictions_before = engine.predict(model=model, dataloaders=loader, return_predictions=True)
    assert predictions_before is not None

    checkpoint_path = tmp_path / "grdnet.ckpt"
    engine.trainer.save_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert len(checkpoint["optimizer_states"]) == 3
    assert len(checkpoint["lr_schedulers"]) == 1

    loaded = GRDNet.load_from_checkpoint(
        checkpoint_path,
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    prediction_engine = _engine(tmp_path / "predict")
    predictions_after = prediction_engine.predict(model=loaded, dataloaders=loader, return_predictions=True)
    assert predictions_after is not None
    assert torch.equal(predictions_before[0].pred_score, predictions_after[0].pred_score)
    assert torch.equal(predictions_before[0].anomaly_map, predictions_after[0].anomaly_map)

    resumed = GRDNet(
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    resume_engine = _engine(tmp_path / "resume")
    resume_engine.fit(
        model=resumed,
        train_dataloaders=loader,
        val_dataloaders=loader,
        ckpt_path=checkpoint_path,
    )
    _assert_nested_equal(
        [optimizer.state_dict() for optimizer in resume_engine.trainer.optimizers],
        checkpoint["optimizer_states"],
    )
    _assert_nested_equal(
        resume_engine.trainer.lr_scheduler_configs[0].scheduler.state_dict(),
        checkpoint["lr_schedulers"][0],
    )


def test_normalizing_preprocessor_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """GRD-Net rejects preprocessing normalization before training."""
    monkeypatch.setattr(GRDNet, "configure_torch_model", staticmethod(_TinyModel))
    preprocessor = PreProcessor(
        Compose(
            [
                Resize((256, 256)),
                Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ],
        ),
    )
    normalized_model = GRDNet(
        pre_processor=preprocessor,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    with pytest.raises(ValueError, match="must not contain Normalize"):
        normalized_model.on_train_start()
