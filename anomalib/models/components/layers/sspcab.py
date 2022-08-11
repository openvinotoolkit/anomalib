"""Self-Supervised Predictive Convolutional Attention (SSPCAB) block for reconstruction-based models."""

# Original Code
# Catalin Ristea
# https://github.com/ristea/sspcab.
# This code is released under the CC BY-SA 4.0 license.

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# Squeeze and Excitation block
class SELayer(nn.Module):
    """Squeeze and excitation layer for the SSPCAB block.

    Args:
        num_channels (int): The number of input channels
        reduction_ratio (int): The reduction ratio 'r' from the paper
    """

    def __init__(self, num_channels: int, reduction_ratio: int = 8):
        super().__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor) -> Tensor:
        """Forward pass through the SE layer."""
        batch_size, num_channels, _, _ = input_tensor.size()

        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        height, width = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(height, width, 1, 1))
        return output_tensor


# SSPCAB implementation
class SSPCAB(nn.Module):
    """SSPCAB block.

    Args:
        channels (int): The number of filter at the output (usually the same with the number of filter from the input)
        kernel_dim (int): The dimension of the sub-kernels ' k' ' from the paper
        dilation (int): The dilation dimension 'd' from the paper
        reduction_ratio (int): The reduction ratio for the SE block ('r' from the paper)
    """

    def __init__(self, channels: int, kernel_dim: int = 1, dilation: int = 1, reduction_ratio: int = 8):
        super().__init__()
        self.pad = kernel_dim + dilation
        self.border_input = kernel_dim + 2 * dilation + 1

        self.relu = nn.ReLU()
        self.se_layer = SELayer(channels, reduction_ratio=reduction_ratio)

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_dim)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_dim)
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_dim)
        self.conv4 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_dim)

    def forward(self, inputs) -> Tensor:
        """Forward pass through the SSPCAB block."""
        inputs = F.pad(inputs, (self.pad, self.pad, self.pad, self.pad), "constant", 0)

        act1 = self.conv1(inputs[:, :, : -self.border_input, : -self.border_input])
        act2 = self.conv2(inputs[:, :, self.border_input :, : -self.border_input])
        act3 = self.conv3(inputs[:, :, : -self.border_input, self.border_input :])
        act4 = self.conv4(inputs[:, :, self.border_input :, self.border_input :])
        act5 = self.relu(act1 + act2 + act3 + act4)

        outputs = self.se_layer(act5)
        return outputs


# Example of how our block should be updated
# mse_loss = nn.MSELoss()
# cost_sspcab = mse_loss(input_sspcab, output_sspcab)
