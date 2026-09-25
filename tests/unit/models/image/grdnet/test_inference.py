# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for GRD-Net patch inference."""

from unittest.mock import patch

import torch
from torch import nn

from anomalib.data import InferenceBatch
from anomalib.models.image.grdnet.torch_model import GRDNetModel


class _Reconstruction(nn.Module):
    """Produce a deterministic reconstruction without a second encoder pass."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def reconstruct(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls += 1
        return torch.empty(0, device=inputs.device), inputs * 0.5


class _DifferenceSegmentator(nn.Module):
    """Derive two-class logits independently for each tile."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        images, reconstructions = inputs.chunk(2, dim=1)
        anomaly_logits = (images - reconstructions).abs().mean(dim=1, keepdim=True)
        return torch.cat((torch.zeros_like(anomaly_logits), anomaly_logits), dim=1)


class _IndexedSegmentator(nn.Module):
    """Assign a known constant anomaly probability to each tile."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        tile_count, _, height, width = inputs.shape
        anomaly_logits = torch.linspace(-1.5, 1.5, tile_count, device=inputs.device, dtype=inputs.dtype)
        anomaly_logits = anomaly_logits.view(-1, 1, 1, 1).expand(-1, 1, height, width)
        return torch.cat((torch.zeros_like(anomaly_logits), anomaly_logits), dim=1)


def _build_model(segmentator: nn.Module) -> GRDNetModel:
    """Build the container without allocating the full segmentator for a focused inference test."""
    with patch(
        "anomalib.models.image.grdnet.torch_model.DiscriminativeSubNetwork",
        return_value=nn.Identity(),
    ):
        model = GRDNetModel(input_size=(128, 128), stride=(64, 64), base_features=1, stage_blocks=(1,))
    model.generator = _Reconstruction()
    model.segmentator = segmentator
    return model.eval()


def test_inference_shapes_finiteness_and_batch_isolation() -> None:
    """Inference should return finite standard outputs without coupling batch items."""
    model = _build_model(_DifferenceSegmentator())
    images = torch.stack((torch.full((3, 256, 256), 0.2), torch.full((3, 256, 256), 0.8)))

    with torch.no_grad():
        predictions = model(images)
        individual = [model(image.unsqueeze(0)) for image in images]

    assert isinstance(predictions, InferenceBatch)
    assert predictions.anomaly_map.shape == (2, 1, 256, 256)
    assert predictions.pred_score.shape == (2,)
    assert torch.isfinite(predictions.anomaly_map).all()
    assert torch.isfinite(predictions.pred_score).all()
    assert torch.allclose(predictions.anomaly_map, torch.cat([item.anomaly_map for item in individual]))
    assert torch.allclose(predictions.pred_score, torch.cat([item.pred_score for item in individual]))


def test_inference_averages_overlaps_smooths_and_scores_non_square_images() -> None:
    """Inference should average overlapping tiles before smoothing and max scoring."""
    model = _build_model(_IndexedSegmentator())
    images = torch.zeros((1, 3, 192, 256))

    with torch.no_grad():
        predictions = model(images)

    probabilities = torch.sigmoid(torch.linspace(-1.5, 1.5, 6))
    expected_map = torch.zeros((1, 1, 192, 256))
    overlap_count = torch.zeros_like(expected_map)
    for probability, top, left in zip(
        probabilities,
        (0, 0, 0, 64, 64, 64),
        (0, 64, 128, 0, 64, 128),
        strict=True,
    ):
        expected_map[..., top : top + 128, left : left + 128] += probability
        overlap_count[..., top : top + 128, left : left + 128] += 1
    expected_map = expected_map / overlap_count
    expected_map = torch.nn.functional.avg_pool2d(expected_map, kernel_size=21, stride=1, padding=10)

    assert predictions.anomaly_map.shape == (1, 1, 192, 256)
    assert torch.allclose(predictions.anomaly_map, expected_map)
    assert torch.equal(predictions.pred_score, expected_map.amax(dim=(-2, -1)).squeeze(1))
