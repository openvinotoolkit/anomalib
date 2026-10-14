# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RAD torch model."""

from typing import Any

import pytest
import torch

from anomalib.models.image.rad.torch_model import RadModel

BACKBONE = "vit_small_patch16_dinov3"


def _build_model(**kwargs: Any) -> RadModel:  # noqa: ANN401
    """Create a small RAD model, overriding the fast-test defaults with ``kwargs``."""
    defaults: dict[str, Any] = {
        "backbone": BACKBONE,
        "pre_trained": False,
        "layers": [3, 11],
        "k_image": 2,
    }
    return RadModel(**(defaults | kwargs))


def _fitted_model(image_size: tuple[int, int] = (224, 224), **kwargs: Any) -> RadModel:  # noqa: ANN401
    """Create a small RAD model with a memory bank built from random images."""
    model = _build_model(**kwargs)
    model.train()
    model(torch.randn(2, 3, *image_size))
    model.build_memory_bank()
    model.eval()
    return model


@pytest.fixture(scope="module")
def model() -> RadModel:
    """Create a RadModel with small settings for fast testing."""
    m = RadModel(
        backbone="vit_small_patch16_dinov3",
        pre_trained=False,
        layers=[3, 6, 9, 11],
        k_image=2,
        use_positional_bank=True,
        pos_radius=1,
        max_ratio=0.01,
    )
    # Build a small memory bank
    m.train()
    x = torch.randn(4, 3, 224, 224)
    m(x)
    m.build_memory_bank()
    return m


@pytest.fixture(scope="module")
def input_tensor() -> torch.Tensor:
    """Create a random input tensor."""
    return torch.randn(2, 3, 224, 224)


def test_initialization(model: RadModel) -> None:
    """Test that the model initialises without errors."""
    assert isinstance(model, RadModel)
    assert len(model.cls_banks) == 4
    assert len(model.patch_banks) == 4


def test_memory_bank_shapes(model: RadModel) -> None:
    """Test that memory bank has correct shapes.

    ``cls_banks`` is ``(layer, image, channel)`` and ``patch_banks`` is position-major,
    ``(layer, position, image, channel)``.
    """
    positions = 224 // model.patch_size * (224 // model.patch_size)
    for i in range(4):
        assert model.cls_banks[i].shape[0] == 4  # 4 training images
        assert model.cls_banks[i].ndim == 2
        assert model.patch_banks[i].ndim == 3
        assert model.patch_banks[i].shape == (positions, 4, model.cls_banks[i].shape[-1])


def test_memory_bank_is_normalized(model: RadModel) -> None:
    """Test the stored banks are unit-norm, as the scoring inner products assume."""
    for bank in (model.cls_banks, model.patch_banks):
        norms = bank.float().norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_fp16_bank_matches_fp32() -> None:
    """Test a half-precision bank halves storage and tracks the full-precision scores."""
    query = torch.randn(1, 3, 224, 224)
    torch.manual_seed(0)
    fp32 = _fitted_model()
    torch.manual_seed(0)
    fp16 = _fitted_model(bank_dtype=torch.float16)

    assert fp16.patch_banks.dtype == torch.float16
    assert fp16.patch_banks.numel() * fp16.patch_banks.element_size() * 2 == (
        fp32.patch_banks.numel() * fp32.patch_banks.element_size()
    )

    with torch.no_grad():
        expected, actual = fp32(query), fp16(query)

    assert torch.allclose(actual.anomaly_map, expected.anomaly_map, atol=1e-2)


def test_forward_eval(model: RadModel, input_tensor: torch.Tensor) -> None:
    """Test forward pass in eval mode returns InferenceBatch."""
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    assert hasattr(output, "anomaly_map")
    assert hasattr(output, "pred_score")
    assert output.anomaly_map.shape[0] == 2
    assert output.anomaly_map.shape[2] == 224
    assert output.anomaly_map.shape[3] == 224
    assert output.pred_score.shape[0] == 2


def test_no_nan_in_output(model: RadModel, input_tensor: torch.Tensor) -> None:
    """Test that outputs contain no NaN values."""
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    assert not torch.isnan(output.anomaly_map).any()
    assert not torch.isnan(output.pred_score).any()


def test_global_matching() -> None:
    """Test model works without positional bank."""
    m = RadModel(
        backbone="vit_small_patch16_dinov3",
        pre_trained=False,
        layers=[3, 11],
        k_image=2,
        use_positional_bank=False,
    )
    m.train()
    m(torch.randn(3, 3, 224, 224))
    m.build_memory_bank()
    m.eval()

    with torch.no_grad():
        output = m(torch.randn(1, 3, 224, 224))

    assert output.pred_score.shape == (1,)
    assert not torch.isnan(output.pred_score).any()


def test_memory_bank_is_checkpointed() -> None:
    """Test the memory bank survives a state dict round trip and reproduces predictions."""
    fitted = _fitted_model()
    state_dict = fitted.state_dict()
    assert "cls_banks" in state_dict
    assert "patch_banks" in state_dict

    restored = _build_model()
    restored.load_state_dict(state_dict)
    restored.eval()

    query = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        expected = fitted(query)
        actual = restored(query)

    assert torch.allclose(actual.pred_score, expected.pred_score)
    assert torch.allclose(actual.anomaly_map, expected.anomaly_map)


def test_scoring_without_memory_bank_raises() -> None:
    """Test inference before fitting fails with an explicit error."""
    model = _build_model()
    model.eval()

    with pytest.raises(ValueError, match="Memory bank is empty"), torch.no_grad():
        model(torch.randn(1, 3, 224, 224))


def test_rectangular_input() -> None:
    """Test non-square images are supported when fit and inference sizes agree."""
    model = _fitted_model(image_size=(224, 336))

    with torch.no_grad():
        output = model(torch.randn(1, 3, 224, 336))

    assert output.anomaly_map.shape == (1, 1, 224, 336)
    assert not torch.isnan(output.anomaly_map).any()


def test_positional_grid_mismatch_raises() -> None:
    """Test position-aware matching rejects a query grid that differs from the bank grid."""
    model = _fitted_model()

    with pytest.raises(ValueError, match="Position-aware matching"), torch.no_grad():
        model(torch.randn(1, 3, 224, 336))


def test_input_size_must_match_patch_size() -> None:
    """Test image sizes that do not tile into whole patches are rejected."""
    model = _fitted_model()

    with pytest.raises(ValueError, match="divisible by the backbone patch size"), torch.no_grad():
        model(torch.randn(1, 3, 225, 225))


def test_max_ratio_zero_uses_max_pooling() -> None:
    """Test ``max_ratio=0`` scores an image with the maximum anomaly pixel."""
    model = _fitted_model(max_ratio=0)

    with torch.no_grad():
        output = model(torch.randn(1, 3, 224, 224))

    assert torch.allclose(output.pred_score, output.anomaly_map.flatten(1).amax(dim=1))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"layers": []}, "at least one transformer block"),
        ({"k_image": 0}, "must be a positive integer"),
        ({"pos_radius": -1}, "must be non-negative"),
        ({"max_ratio": 1.5}, r"must lie in the range \[0, 1\]"),
        ({"layer_weights": [1.0]}, "one entry per layer"),
        ({"layer_weights": [1.0, -1.0]}, "must be non-negative"),
        ({"layer_weights": [0.0, 0.0]}, "must sum to a positive value"),
    ],
)
def test_invalid_arguments_raise(kwargs: dict[str, Any], message: str) -> None:
    """Test invalid constructor arguments fail fast with an actionable message."""
    with pytest.raises(ValueError, match=message):
        _build_model(**kwargs)
