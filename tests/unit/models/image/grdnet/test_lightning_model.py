# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GRD-Net Lightning lifecycle."""

from copy import deepcopy
from pathlib import Path
from typing import NamedTuple
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import CenterCrop, Compose, Normalize, RandomHorizontalFlip, Resize

from anomalib import LearningType
from anomalib.data import ImageBatch, ImageItem, InferenceBatch
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.engine.plugins import AnomalibCheckpointIO
from anomalib.models.components import AnomalibModule
from anomalib.models.image.grdnet import lightning_model
from anomalib.models.image.grdnet.data.dataclasses import GRDNetBatch, GRDNetItem
from anomalib.models.image.grdnet.lightning_model import GRDNet
from anomalib.models.image.grdnet.loss import GeneratorLosses
from anomalib.models.image.grdnet.pre_processor import GRDNetPreProcessor
from anomalib.models.image.grdnet.torch_model import GRDNetModel
from anomalib.pre_processing import PreProcessor


class _TinyGenerator(nn.Module):
    """Small generator with batch normalization for optimizer-boundary tests."""

    def __init__(self) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(3, 3, kernel_size=1)
        self.normalization = nn.BatchNorm2d(3)
        self.last_input: torch.Tensor | None = None

    def reconstruct(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a compact latent and bounded reconstruction."""
        self.last_input = images.detach().clone()
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
        self.image_input: torch.Tensor | None = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return two-class logits and record the reconstruction gradient flag."""
        self.image_input = images[:, :3].detach().clone()
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


def _tiny_inference_model() -> GRDNetModel:
    """Keep real tiling, pooling, and scoring while substituting small local subnetworks."""
    with patch(
        "anomalib.models.image.grdnet.torch_model.DiscriminativeSubNetwork",
        return_value=_TinySegmentator(),
    ):
        container = GRDNetModel(base_features=1, stage_blocks=(1,))
    container.generator = _TinyGenerator()
    container.discriminator = _TinyDiscriminator()
    return container


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


@pytest.mark.parametrize(
    "argument",
    [
        "contextual_l1_weight",
        "contextual_ssim_weight",
        "generator_scheduler_decay",
        "generator_scheduler_patience",
        "rotation_degrees",
        "smoothing_kernel_size",
    ],
)
@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -1])
def test_new_parameters_reject_invalid_numeric_values(argument: str, value: object) -> None:
    """All new public controls reject booleans, nonfinite values, and negative numbers before construction."""
    with pytest.raises(ValueError, match=argument):
        GRDNet(**{argument: value})


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("generator_scheduler_decay", 0.0),
        ("generator_scheduler_decay", 1e-20),
        ("generator_scheduler_decay", 1000.0),
        ("generator_scheduler_patience", 1.0),
        ("smoothing_kernel_size", 3.0),
        ("smoothing_kernel_size", 0),
        ("smoothing_kernel_size", 2),
    ],
)
def test_new_parameter_boundaries(argument: str, value: object) -> None:
    """Scheduler factors must reduce the rate and integer controls must retain their exact type."""
    with pytest.raises(ValueError, match=argument):
        GRDNet(**{argument: value})


def test_both_inner_contextual_coefficients_cannot_be_zero() -> None:
    """A degenerate contextual objective is rejected even when the outer coefficient is zero."""
    with pytest.raises(ValueError, match="contextual_l1_weight and contextual_ssim_weight"):
        GRDNet(contextual_l1_weight=0.0, contextual_ssim_weight=0.0, contextual_weight=0.0)


@pytest.mark.parametrize(("l1_weight", "ssim_weight"), [(0.0, 1.0), (1.0, 0.0)])
def test_zero_parameter_boundaries_and_positional_components(
    model: GRDNet,
    l1_weight: float,
    ssim_weight: float,
) -> None:
    """Valid zero controls do not change existing positional component arguments or tensor state."""
    component_flags = (False, False, False, False)
    configured = GRDNet(
        "random",
        0.75,
        1.0,
        0.0,
        1.0,
        1e-4,
        *component_flags,
        contextual_l1_weight=l1_weight,
        contextual_ssim_weight=ssim_weight,
        generator_scheduler_patience=0,
        rotation_degrees=0.0,
        smoothing_kernel_size=1,
    )
    assert configured.pre_processor is None
    assert configured.post_processor is None
    assert configured.evaluator is None
    assert configured.visualizer is None
    assert configured.generator_loss.contextual_l1_weight == l1_weight
    assert configured.generator_loss.contextual_ssim_weight == ssim_weight
    assert configured.generator_scheduler_patience == 0
    assert configured.rotation_degrees == 0
    assert configured.model.smoothing_kernel_size == 1
    assert configured.state_dict().keys() == model.state_dict().keys()


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
    perturbed = torch.rand_like(images)
    before = {name: _parameter_state(module) for name, module in model.model.named_children()}

    model._update_discriminator(images, perturbed, optimizer_config.optimizers[0])  # noqa: SLF001

    assert _state_changed(before["discriminator"], model.model.discriminator)
    assert not _state_changed(before["generator"], model.model.generator)
    assert not _state_changed(before["segmentator"], model.model.segmentator)
    assert torch.equal(model.model.generator.last_input, perturbed)


def test_generator_phase_freezes_discriminator(
    model: GRDNet,
    optimizer_config: _OptimizerConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second phase updates only the generator and freezes discriminator BN state."""
    images = torch.rand(2, 3, 8, 8)
    perturbed = torch.rand_like(images)
    before = {name: _parameter_state(module) for name, module in model.model.named_children()}
    running_mean = model.model.discriminator.normalization.running_mean.clone()
    clean_targets: list[torch.Tensor] = []
    generator_loss = model.generator_loss.forward

    def capture_clean_target(*args: torch.Tensor) -> GeneratorLosses:
        clean_targets.append(args[0].detach().clone())
        return generator_loss(*args)

    monkeypatch.setattr(model.generator_loss, "forward", capture_clean_target)

    model._update_generator(images, perturbed, optimizer_config.optimizers[1])  # noqa: SLF001

    assert _state_changed(before["generator"], model.model.generator)
    assert not _state_changed(before["discriminator"], model.model.discriminator)
    assert not _state_changed(before["segmentator"], model.model.segmentator)
    assert torch.equal(running_mean, model.model.discriminator.normalization.running_mean)
    assert all(parameter.requires_grad for parameter in model.model.discriminator.parameters())
    assert torch.equal(model.model.generator.last_input, perturbed)
    assert len(clean_targets) == 1
    assert torch.equal(clean_targets[0], images)


@pytest.mark.parametrize("optimizer_index", [0, 1, 2])
def test_gradient_clipping_applies_only_to_generator(
    model: GRDNet,
    optimizer_config: _OptimizerConfig,
    monkeypatch: pytest.MonkeyPatch,
    optimizer_index: int,
) -> None:
    """Only the generator optimizer clips gradients, leaving other phases unchanged."""
    modules = (model.model.discriminator, model.model.generator, model.model.segmentator)
    parameters = [next(module.parameters()) for module in modules]
    for parameter in parameters:
        parameter.grad = torch.full_like(parameter, 2.0)
    original_gradients = [parameter.grad.clone() for parameter in parameters]
    monkeypatch.setattr(model, "optimizers", MagicMock(return_value=optimizer_config.optimizers))

    model.on_before_optimizer_step(optimizer_config.optimizers[optimizer_index])

    for index, parameter in enumerate(parameters):
        if optimizer_index == 1 and index == 1:
            assert parameter.grad.norm().item() == pytest.approx(1.0)
        else:
            torch.testing.assert_close(parameter.grad, original_gradients[index])


@pytest.mark.parametrize("scale", [1.0, 65536.0])
@pytest.mark.parametrize("gradient_norm", [0.5, 5.0])
def test_generator_clipping_occurs_after_unscaling(
    model: GRDNet,
    optimizer_config: _OptimizerConfig,
    monkeypatch: pytest.MonkeyPatch,
    scale: float,
    gradient_norm: float,
) -> None:
    """Generator clipping preserves its norm threshold regardless of loss scaling."""
    optimizer = optimizer_config.optimizers[1]
    scaler = torch.amp.GradScaler("cpu", init_scale=scale)
    parameter = next(model.model.generator.parameters())
    observed_norms: list[float] = []
    original_step = optimizer.step

    def controlled_loss(*args: torch.Tensor) -> GeneratorLosses:
        """Give one generator parameter a known gradient."""
        del args
        loss = parameter.flatten()[0] * gradient_norm
        return GeneratorLosses(loss, loss, loss, loss)

    def optimizer_step() -> None:
        """Unscale before the pre-step hook, matching Lightning's precision boundary."""
        scaler.unscale_(optimizer)
        model.on_before_optimizer_step(optimizer)
        observed_norms.append(parameter.grad.norm().item())
        original_step()

    monkeypatch.setattr(model.generator_loss, "forward", controlled_loss)
    monkeypatch.setattr(model, "manual_backward", lambda loss: scaler.scale(loss).backward())
    monkeypatch.setattr(model, "optimizers", MagicMock(return_value=optimizer_config.optimizers))
    monkeypatch.setattr(optimizer, "step", optimizer_step)
    images = torch.linspace(0.0, 1.0, 2 * 3 * 8 * 8).reshape(2, 3, 8, 8)

    model._update_generator(images, images, optimizer)  # noqa: SLF001

    assert observed_norms == pytest.approx([min(gradient_norm, 1.0)])


def test_segmentator_phase_uses_detached_reconstruction(
    model: GRDNet,
    optimizer_config: _OptimizerConfig,
) -> None:
    """The third phase updates only the segmentator with a detached reconstruction."""
    images = torch.rand(2, 3, 8, 8)
    perturbed = torch.rand_like(images)
    masks = torch.ones(2, 1, 8, 8)
    before = {name: _parameter_state(module) for name, module in model.model.named_children()}

    model._update_segmentator(perturbed, masks, masks, optimizer_config.optimizers[2])  # noqa: SLF001

    assert _state_changed(before["segmentator"], model.model.segmentator)
    assert not _state_changed(before["generator"], model.model.generator)
    assert not _state_changed(before["discriminator"], model.model.discriminator)
    assert model.model.segmentator.reconstruction_requires_grad is False
    assert torch.equal(model.model.segmentator.image_input, perturbed)
    assert torch.equal(model.model.generator.last_input, perturbed)


def test_standard_batch_uses_full_roi() -> None:
    """Standard image batches receive an all-ones training ROI."""
    batch = ImageBatch(image=torch.rand(2, 3, 7, 9))
    roi_masks = GRDNet._roi_masks(batch)  # noqa: SLF001
    assert roi_masks.shape == (2, 1, 7, 9)
    assert roi_masks.dtype == batch.image.dtype
    assert roi_masks.all()


def test_training_uses_model_geometry_and_rotation(
    model: GRDNet,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Training uses the Torch model's geometry and forwards the configured rotation bound."""
    model.model.tile_size = (256, 256)
    model.model.stride = (256, 256)
    model.rotation_degrees = 30.0
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

    def capture_rotation(
        images: torch.Tensor,
        roi_masks: torch.Tensor,
        *,
        rotation_degrees: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        captured["rotation_degrees"] = rotation_degrees
        return images, roi_masks

    loss = torch.tensor(1.0)
    generator_losses = GeneratorLosses(loss, loss, loss, loss)
    monkeypatch.setattr(lightning_model, "tile_image_and_roi", capture_geometry)
    monkeypatch.setattr(lightning_model, "rotate_image_and_roi", capture_rotation)
    monkeypatch.setattr(model, "optimizers", lambda: (MagicMock(), MagicMock(), MagicMock()))
    monkeypatch.setattr(model, "_update_discriminator", MagicMock(return_value=loss))
    monkeypatch.setattr(model, "_update_generator", MagicMock(return_value=generator_losses))
    monkeypatch.setattr(model, "_update_segmentator", MagicMock(return_value=loss))
    monkeypatch.setattr(model, "log_dict", MagicMock())

    model.training_step(ImageBatch(image=torch.rand(1, 3, 256, 256)), 0)

    assert captured == {"tile_size": (256, 256), "stride": (256, 256), "tile_count": 1, "rotation_degrees": 30.0}


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


@pytest.mark.usefixtures("model")
def test_nondefault_scheduler_monitors_inner_weighted_contextual_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scheduler uses inner-weighted contextual loss without its outer coefficient and changes only generator LR."""
    configured = GRDNet(
        contextual_weight=0.0,
        contextual_l1_weight=2.0,
        contextual_ssim_weight=0.5,
        generator_scheduler_decay=0.2,
        generator_scheduler_patience=0,
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    optimizers, schedulers = configured.configure_optimizers()
    scheduler = schedulers[0]
    monkeypatch.setattr(configured, "lr_schedulers", lambda: scheduler)
    monkeypatch.setattr(configured, "log", MagicMock())
    inputs = torch.zeros((1, 3, 11, 11))
    reconstruction = torch.ones_like(inputs)
    losses = configured.generator_loss(inputs, reconstruction, inputs, inputs, inputs, inputs)
    contextual = losses.contextual.detach()
    assert losses.total == 0
    for _ in range(2):
        configured.contextual_loss_mean.update(contextual)
        configured.contextual_loss_mean.update(contextual * 3)
        configured.on_train_epoch_end()
    assert scheduler.best == pytest.approx((2 * contextual).item())
    assert scheduler.patience == 0
    assert optimizers[0].param_groups[0]["lr"] == configured.learning_rate
    assert optimizers[1].param_groups[0]["lr"] == pytest.approx(configured.learning_rate * scheduler.factor)
    assert scheduler.factor == pytest.approx(torch.exp(torch.tensor(-0.2)).item())
    assert optimizers[2].param_groups[0]["lr"] == configured.learning_rate
    assert configured.contextual_loss_mean.update_count == 0


def test_validation_step_returns_standard_predictions(model: GRDNet) -> None:
    """Validation merges standard GRD-Net predictions into the input batch."""
    batch = ImageBatch(image=torch.rand(2, 3, 8, 10))
    result = model.validation_step(deepcopy(batch))
    assert result.pred_score.shape == (2,)
    assert result.anomaly_map.shape == (2, 8, 10)


@pytest.mark.parametrize(
    ("input_size", "options"),
    [
        ((256, 256), {}),
        (None, {}),
        ((256, 256), {"dynamic_shapes": None}),
        ((256, 256), {"dynamic_shapes": ({0: torch.export.Dim.DYNAMIC},)}),
        ((256, 256), {"dynamic_axes": {"input": {0: "batch"}}}),
    ],
)
def test_onnx_shape_options_preserve_caller_settings(
    model: GRDNet,
    tmp_path: Path,
    input_size: tuple[int, int] | None,
    options: dict[str, object],
) -> None:
    """Only default fixed-spatial exports receive the GRD-Net dynamic-batch hint."""
    with patch.object(AnomalibModule, "to_onnx", return_value=tmp_path / "custom.onnx") as export:
        result = model.to_onnx(tmp_path, "custom", input_size, opset_version=18, **options)

    expected = dict(options)
    if input_size is not None and not options:
        expected["dynamic_shapes"] = ({0: torch.export.Dim.DYNAMIC},)
    export.assert_called_once_with(tmp_path, "custom", input_size, opset_version=18, **expected)
    assert result == tmp_path / "custom.onnx"


@pytest.mark.usefixtures("model")
@pytest.mark.parametrize("export_type", [ExportType.ONNX, ExportType.OPENVINO])
def test_dynamic_batch_export_matches_eager(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    export_type: ExportType,
) -> None:
    """Real tile/stitch inference stays dynamic and numerically equivalent after export."""
    import numpy as np

    pytest.importorskip("onnxscript")
    onnx = pytest.importorskip("onnx")
    if export_type == ExportType.OPENVINO:
        openvino = pytest.importorskip("openvino")
    monkeypatch.setattr(GRDNet, "configure_torch_model", staticmethod(_tiny_inference_model))
    configured = GRDNet(pre_processor=False, post_processor=False, evaluator=False, visualizer=False).eval()
    if export_type == ExportType.ONNX:
        from onnx.reference import ReferenceEvaluator

        path = configured.to_onnx(tmp_path, input_size=(256, 256))
        exported = onnx.load(path)
        assert exported.graph.input[0].type.tensor_type.shape.dim[0].dim_param
        runtime = ReferenceEvaluator(exported)

        def predict(images: np.ndarray) -> dict[str, np.ndarray]:
            return dict(
                zip(
                    runtime.output_names,
                    runtime.run(None, {runtime.input_names[0]: images}),
                    strict=True,
                ),
            )

    else:
        path = configured.to_openvino(tmp_path, input_size=(256, 256))
        compiled = openvino.Core().compile_model(str(path), "CPU")

        def predict(images: np.ndarray) -> dict[str, np.ndarray]:
            outputs = compiled(images)
            return {output.get_any_name(): outputs[output] for output in compiled.outputs}

    for size in (1, 2):
        images = torch.linspace(0, 1, size * 3 * 256 * 256).reshape(size, 3, 256, 256)
        with torch.no_grad():
            expected = configured(images)
        actual = predict(images.numpy())
        np.testing.assert_allclose(actual["pred_score"], expected.pred_score.numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(actual["anomaly_map"], expected.anomaly_map.numpy(), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "hook_name",
    ["on_train_batch_start", "on_validation_batch_start", "on_test_batch_start", "on_predict_batch_start"],
)
def test_roi_guard_rejects_incompatible_processor_before_mutation(model: GRDNet, hook_name: str) -> None:
    """An ordinary processor cannot silently misalign shape-preserving ROI supervision."""
    processor = PreProcessor(RandomHorizontalFlip(p=1))
    model.pre_processor = processor
    images = torch.arange(3 * 8 * 10).float().reshape(1, 3, 8, 10) / 240
    batch = GRDNetBatch(image=images.clone(), roi_mask=torch.ones(1, 8, 10))
    callbacks = model.configure_callbacks()
    assert callbacks.index(processor) > 0

    with pytest.raises(ValueError, match="GRDNetPreProcessor"):
        getattr(callbacks[0], hook_name)(None, model, batch, 0)

    assert torch.equal(batch.image, images)


def test_roi_guard_preserves_ordinary_custom_processor_without_roi(model: GRDNet) -> None:
    """Custom processor instances retain their behavior for standard batches."""
    processor = PreProcessor(RandomHorizontalFlip(p=1))
    model.pre_processor = processor
    batch = ImageBatch(image=torch.rand(1, 3, 8, 10))
    expected = batch.image.flip(-1)
    for callback in model.configure_callbacks():
        callback.on_train_batch_start(None, model, batch, 0)
    assert model.pre_processor is processor
    assert torch.equal(batch.image, expected)


@pytest.mark.parametrize("enabled", [False, True])
def test_roi_guard_accepts_disabled_or_joint_preprocessing(model: GRDNet, enabled: bool) -> None:
    """Already aligned inputs and subclasses of the joint processor remain supported."""

    class JointProcessor(GRDNetPreProcessor):
        """User specialization retaining joint transform behavior."""

    model.pre_processor = JointProcessor(RandomHorizontalFlip(p=1)) if enabled else None
    batch = GRDNetBatch(image=torch.rand(1, 3, 8, 10), roi_mask=torch.ones(1, 8, 10))
    expected = batch.image.flip(-1) if enabled else batch.image.clone()
    for callback in model.configure_callbacks():
        callback.on_train_batch_start(None, model, batch, 0)
    assert torch.equal(batch.image, expected)


def test_custom_joint_processor_checkpoint_roundtrip(model: GRDNet, tmp_path: Path) -> None:
    """A deterministic custom joint transform survives the model checkpoint path."""
    model.pre_processor = GRDNetPreProcessor(Compose([CenterCrop((128, 128)), Resize((256, 256))]))
    loader = _loader(with_roi=True)
    engine = _engine(tmp_path / "fit")
    engine.fit(model=model, train_dataloaders=loader, val_dataloaders=loader)
    path = tmp_path / "joint.ckpt"
    engine.trainer.save_checkpoint(path)
    loaded = GRDNet.load_from_checkpoint(path, post_processor=False, evaluator=False, visualizer=False)
    mask = torch.zeros(1, 256, 256)
    mask[..., 32:128, 64:192] = 1
    original_batch = GRDNetBatch(image=mask.unsqueeze(1).repeat(1, 3, 1, 1), roi_mask=mask)
    restored_batch = deepcopy(original_batch)
    model.pre_processor.on_predict_batch_start(None, model, original_batch, 0)
    loaded.pre_processor.on_predict_batch_start(None, loaded, restored_batch, 0)
    assert isinstance(loaded.pre_processor, GRDNetPreProcessor)
    assert torch.equal(original_batch.image, restored_batch.image)
    assert torch.equal(original_batch.roi_mask, restored_batch.roi_mask)


def _loader(*, with_roi: bool) -> DataLoader:
    """Create one deterministic image or ROI-aware batch."""
    image = torch.linspace(0, 1, 3 * 256 * 256).reshape(3, 256, 256)
    if with_roi:
        item = GRDNetItem(image=image, roi_mask=torch.ones(256, 256))
        collate_fn = GRDNetBatch.collate
    else:
        item = ImageItem(image=image)
        collate_fn = ImageBatch.collate
    # This CPU-only fixture has no accelerator transfer requiring pinned memory.
    return DataLoader(  # nosemgrep: trailofbits.python.automatic-memory-pinning.automatic-memory-pinning
        [item],
        batch_size=1,
        collate_fn=collate_fn,
        pin_memory=False,
    )


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


@pytest.mark.usefixtures("model")
@pytest.mark.parametrize(
    "parameters",
    [
        {},
        {
            "contextual_l1_weight": 2.0,
            "contextual_ssim_weight": 0.5,
            "generator_scheduler_decay": 0.2,
            "generator_scheduler_patience": 0,
            "rotation_degrees": 30.0,
            "smoothing_kernel_size": 3,
        },
    ],
)
def test_checkpoint_roundtrip_restores_training_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parameters: dict[str, float | int],
) -> None:
    """Engine checkpoints restore networks, optimizers, scheduler, and predictions."""
    monkeypatch.setattr(GRDNet, "configure_torch_model", staticmethod(_tiny_inference_model))
    model = GRDNet(**parameters, pre_processor=False, post_processor=False, evaluator=False, visualizer=False)
    loader = _loader(with_roi=False)
    engine = _engine(tmp_path / "fit")
    engine.fit(model=model, train_dataloaders=loader, val_dataloaders=loader)
    predictions_before = engine.predict(model=model, dataloaders=loader, return_predictions=True)
    assert predictions_before is not None

    checkpoint_path = tmp_path / "grdnet.ckpt"
    engine.trainer.save_checkpoint(checkpoint_path)
    checkpoint_io = AnomalibCheckpointIO(extra_safe_globals=model.checkpoint_safe_globals())
    checkpoint = checkpoint_io.load_checkpoint(checkpoint_path, map_location="cpu")
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
    assert loaded.hparams == model.hparams
    assert loaded.model.smoothing_kernel_size == model.model.smoothing_kernel_size
    assert loaded.rotation_degrees == model.rotation_degrees
    inputs = torch.linspace(0, 1, 3 * 11 * 11).reshape(1, 3, 11, 11)
    reconstruction = inputs * 0.5
    expected_losses = model.generator_loss(inputs, reconstruction, inputs, inputs, inputs, inputs)
    actual_losses = loaded.generator_loss(inputs, reconstruction, inputs, inputs, inputs, inputs)
    for actual, expected in zip(actual_losses, expected_losses, strict=True):
        torch.testing.assert_close(actual, expected)
    _, loaded_schedulers = loaded.configure_optimizers()
    assert loaded_schedulers[0].factor == engine.trainer.lr_scheduler_configs[0].scheduler.factor
    assert loaded_schedulers[0].patience == engine.trainer.lr_scheduler_configs[0].scheduler.patience

    if parameters:
        legacy_checkpoint = deepcopy(checkpoint)
        for name in parameters:
            del legacy_checkpoint["hyper_parameters"][name]
        legacy_path = tmp_path / "legacy.ckpt"
        checkpoint_io.save_checkpoint(legacy_checkpoint, legacy_path)
        legacy = GRDNet.load_from_checkpoint(legacy_path, evaluator=False, visualizer=False)
        assert legacy.generator_loss.contextual_l1_weight == 1.0
        assert legacy.generator_loss.contextual_ssim_weight == 1.0
        assert legacy.generator_scheduler_decay == 0.1
        assert legacy.generator_scheduler_patience == 3
        assert legacy.rotation_degrees == 90.0
        assert legacy.model.smoothing_kernel_size == 21
        _assert_nested_equal(legacy.state_dict(), loaded.state_dict())

    resumed = GRDNet(
        **parameters,
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


@pytest.mark.usefixtures("model")
def test_nondefault_smoothing_torch_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Torch export retains nondefault pooling and agrees with eager image-only inference."""
    monkeypatch.setattr(GRDNet, "configure_torch_model", staticmethod(_tiny_inference_model))
    configured = GRDNet(
        smoothing_kernel_size=3,
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    ).eval()
    images = torch.linspace(0, 1, 3 * 256 * 256).reshape(1, 3, 256, 256)
    export_path = _engine(tmp_path / "runtime").export(
        model=configured,
        export_type=ExportType.TORCH,
        export_root=tmp_path / "export",
    )
    assert export_path is not None
    # Full-object export is trusted here: this test just generated the temporary artifact.
    # nosemgrep: trailofbits.python.pickles-in-pytorch.pickles-in-pytorch
    exported = torch.load(export_path, map_location="cpu", weights_only=False)["model"].eval()
    assert exported.model.smoothing_kernel_size == 3
    with torch.no_grad():
        expected = configured(images)
        actual = exported(images)
    torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.pred_score, expected.pred_score, rtol=1e-5, atol=1e-6)


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
