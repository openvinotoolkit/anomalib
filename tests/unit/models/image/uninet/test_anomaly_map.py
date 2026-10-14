# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for UniNet's weighted decision mechanism."""

import pytest
import torch
from tests.helpers.export import assert_exportable

from anomalib.models.components.filters import GaussianBlur2d
from anomalib.models.image.uninet.components import weighted_decision_mechanism


def _reference_score(output_list: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
    """Per-image score as in the original loop: max of the blurred, summed map."""
    blur = GaussianBlur2d(sigma=4.0, kernel_size=(5, 5), channels=1)
    summed = sum(
        torch.nn.functional.interpolate(o.unsqueeze(1), output_size, mode="bilinear", align_corners=True).squeeze(1)
        for o in output_list
    )
    return torch.stack([blur(image[None, None]).max() for image in summed]).unsqueeze(1)


def _score_two_outputs(first: torch.Tensor, second: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Score two output maps; tensor-only signature for ``assert_exportable``."""
    return weighted_decision_mechanism(first.shape[0], [first, second], 0.01, 3e-05, (64, 64))


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("output_size", [(64, 64), (48, 80)])
def test_score_is_max_of_blurred_map(batch_size: int, output_size: tuple[int, int]) -> None:
    """Score equals the max of the blurred map; map is the resized sum of outputs."""
    torch.manual_seed(0)
    outputs = [torch.rand(batch_size, 32, 32) * 2, torch.rand(batch_size, 16, 16) * 2]

    score, anomaly_map = weighted_decision_mechanism(batch_size, outputs, 0.01, 3e-05, output_size)

    assert score.shape == (batch_size, 1)
    assert anomaly_map.shape == (batch_size, *output_size)
    torch.testing.assert_close(score, _reference_score(outputs, output_size))


def test_weighted_decision_mechanism_is_exportable() -> None:
    """``torch.export`` captures the scoring path without data-dependent guards and matches eager."""
    assert_exportable(_score_two_outputs, torch.rand(2, 32, 32), torch.rand(2, 16, 16))
