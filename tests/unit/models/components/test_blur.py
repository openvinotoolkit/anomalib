"""Test if our implementation produces same result as kornia."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from kornia.filters import GaussianBlur2d as korniaGaussianBlur2d

from anomalib.models.components import GaussianBlur2d


@pytest.mark.parametrize("sigma", [(4.0, 4.0), (1.9, 3.0), (2.0, 1.5)])
@pytest.mark.parametrize("channels", list(range(1, 6)))
@pytest.mark.parametrize("kernel_size", [(33, 33), (9, 9), (11, 5), (3, 3)])
def test_blur_equivalence(kernel_size: tuple[int, int], sigma: tuple[int, int], channels: int) -> None:
    """Test if the GaussianBlur2d layer produces the same output as Kornia's GaussianBlur2d layer."""
    for _ in range(10):
        input_tensor = torch.randn((3, channels, 128, 128))
        kornia = korniaGaussianBlur2d(kernel_size, sigma, separable=False)
        blur_kornia = kornia(input_tensor)
        gaussian = GaussianBlur2d(kernel_size=kernel_size, sigma=sigma, channels=channels)
        blur_gaussian = gaussian(input_tensor)
        torch.testing.assert_close(blur_kornia, blur_gaussian)
