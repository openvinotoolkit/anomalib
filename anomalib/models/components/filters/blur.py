"""Gaussian blurring via pytorch."""

from typing import Tuple, Union

from kornia.filters import get_gaussian_kernel2d
from kornia.filters.filter import _compute_padding
from kornia.filters.kernels import normalize_kernel2d
from torch import Tensor, nn
from torch.nn import functional as F


class GaussianBlur2d(nn.Module):
    """Compute GaussianBlur in 2d.

    Makes use of kornia functions, but most notably the kernel is not computed
    during the forward pass, and does not depend on the input size. As a caveat,
    the number of channels that are expected have to be provided during initialization.
    """

    def __init__(
        self,
        kernel_size: Union[Tuple[int, int], int],
        sigma: Union[Tuple[float, float], float],
        channels: int,
        normalize: bool = True,
        border_type: str = "reflect",
        padding: str = "same",
    ) -> None:
        """Initialize model, setup kernel etc..

        Args:
            kernel_size (Union[Tuple[int, int], int]): size of the Gaussian kernel to use.
            sigma (Union[Tuple[float, float], float]): standard deviation to use for constructing the Gaussian kernel.
            channels (int): channels of the input
            normalize (bool, optional): Whether to normalize the kernel or not (i.e. all elements sum to 1).
                Defaults to True.
            border_type (str, optional): Border type to use for padding of the input. Defaults to "reflect".
            padding (str, optional): Type of padding to apply. Defaults to "same".
        """
        super().__init__()
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        sigma = sigma if isinstance(sigma, tuple) else (sigma, sigma)
        self.kernel: Tensor
        self.register_buffer("kernel", get_gaussian_kernel2d(kernel_size=kernel_size, sigma=sigma))
        if normalize:
            self.kernel = normalize_kernel2d(self.kernel)
        self.channels = channels
        self.kernel.unsqueeze_(0).unsqueeze_(0)
        self.kernel = self.kernel.expand(self.channels, -1, -1, -1)
        self.border_type = border_type
        self.padding = padding
        self.height, self.width = self.kernel.shape[-2:]
        self.padding_shape = _compute_padding([self.height, self.width])

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Blur the input with the computed Gaussian.

        Args:
            input_tensor (Tensor): Input tensor to be blurred.

        Returns:
            Tensor: Blurred output tensor.
        """
        batch, channel, height, width = input_tensor.size()

        if self.padding == "same":
            input_tensor = F.pad(input_tensor, self.padding_shape, mode=self.border_type)

        # convolve the tensor with the kernel.
        output = F.conv2d(input_tensor, self.kernel, groups=self.channels, padding=0, stride=1)

        if self.padding == "same":
            out = output.view(batch, channel, height, width)
        else:
            out = output.view(batch, channel, height - self.height + 1, width - self.width + 1)

        return out
