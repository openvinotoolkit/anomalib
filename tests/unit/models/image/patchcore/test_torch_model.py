# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the PatchCore torch model."""

import pytest
import torch

from anomalib.data.utils.tiler import Tiler
from anomalib.models.image.patchcore import torch_model
from anomalib.models.image.patchcore.torch_model import PatchcoreModel


@pytest.fixture(params=[False, True], ids=["untiled", "tiled"])
def model(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> PatchcoreModel:
    """PatchCore with a random memory bank and a chunk size smaller than one image's patches."""
    # A 64x64 image gives 8x8 = 64 patches, so eager inference searches the memory bank in several chunks.
    monkeypatch.setattr(torch_model, "DEFAULT_CHUNK_SIZE", 16)
    torch.manual_seed(0)
    model = PatchcoreModel(backbone="resnet18", layers=["layer2", "layer3"], pre_trained=False, num_neighbors=3)
    model.memory_bank = torch.randn(32, 384)  # resnet18 layer2 + layer3 channels
    if request.param:
        model.tiler = Tiler(tile_size=32, stride=32)
    return model.eval()


def test_export_keeps_batch_size_dynamic(model: PatchcoreModel) -> None:
    """Export searches all patches at once, so any batch size runs and matches chunked eager inference."""
    batch_size = torch.export.Dim("batch_size")
    exported = torch.export.export(model, (torch.rand(2, 3, 64, 64),), dynamic_shapes=({0: batch_size},)).module()

    for size in (1, 3):
        images = torch.rand(size, 3, 64, 64)
        actual, expected = exported(images), model(images)
        torch.testing.assert_close(actual.pred_score, expected.pred_score)
        torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map)
