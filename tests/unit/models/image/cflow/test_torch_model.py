# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the CFlow torch model."""

from functools import partial

import torch
from tests.helpers.export import assert_exportable

from anomalib.models.image.cflow.torch_model import CflowModel


def _scores(model: CflowModel, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return score and map as tensors for ``assert_exportable``."""
    output = model(images)
    return output.pred_score, output.anomaly_map


def test_export_matches_chunked_eager_inference() -> None:
    """Export decodes all rows at once; eager decodes in fiber batches. Results must match."""
    torch.manual_seed(0)
    # A small fiber batch forces several chunks per layer in eager mode.
    model = CflowModel(backbone="resnet18", layers=["layer1", "layer2"], pre_trained=False, fiber_batch_size=16).eval()
    assert_exportable(partial(_scores, model), torch.rand(2, 3, 64, 64))
