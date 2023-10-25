import pytest
import torch
from kornia.filters import GaussianBlur2d as korniaGaussianBlur2d

from anomalib.models.components import GaussianBlur2d


@pytest.mark.parametrize("sigma", [(4.0, 4.0), (1.9, 3.0), (2.0, 1.5)])
@pytest.mark.parametrize("channels", list(range(1, 6)))
@pytest.mark.parametrize("kernel_size", [(33, 33), (9, 9), (11, 5), (3, 3)])
def test_blur_equivalence(kernel_size, sigma, channels):
    for _ in range(10):
        input_tensor = torch.randn((3, channels, 128, 128))
        kornia = korniaGaussianBlur2d(kernel_size, sigma, separable=False)
        blur_kornia = kornia(input_tensor)
        gaussian = GaussianBlur2d(kernel_size=kernel_size, sigma=sigma, channels=channels)
        blur_gaussian = gaussian(input_tensor)
        torch.testing.assert_allclose(blur_kornia, blur_gaussian)
