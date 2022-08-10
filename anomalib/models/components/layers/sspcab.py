"""SSPCAB block for reconstruction-based models."""

# Original Code
# Catalin Ristea
# https://github.com/ristea/sspcab.
# This code is released under the CC BY-SA 4.0 license.

import torch
import torch.nn as nn
import torch.nn.functional as F


# Squeeze and Excitation block
class SELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=8):
        """
        num_channels: The number of input channels
        reduction_ratio: The reduction ratio 'r' from the paper
        """
        super(SELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()

        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


# SSPCAB implementation
class SSPCAB(nn.Module):
    def __init__(self, channels, kernel_dim=1, dilation=1, reduction_ratio=8):
        """
        channels: The number of filter at the output (usually the same with the number of filter from the input)
        kernel_dim: The dimension of the sub-kernels ' k' ' from the paper
        dilation: The dilation dimension 'd' from the paper
        reduction_ratio: The reduction ratio for the SE block ('r' from the paper)
        """
        super(SSPCAB, self).__init__()
        self.pad = kernel_dim + dilation
        self.border_input = kernel_dim + 2 * dilation + 1

        self.relu = nn.ReLU()
        self.se = SELayer(channels, reduction_ratio=reduction_ratio)

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_dim)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_dim)
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_dim)
        self.conv4 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_dim)

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), "constant", 0)

        x1 = self.conv1(x[:, :, : -self.border_input, : -self.border_input])
        x2 = self.conv2(x[:, :, self.border_input :, : -self.border_input])
        x3 = self.conv3(x[:, :, : -self.border_input, self.border_input :])
        x4 = self.conv4(x[:, :, self.border_input :, self.border_input :])
        x = self.relu(x1 + x2 + x3 + x4)

        x = self.se(x)
        return x


# Example of how our block should be updated
# mse_loss = nn.MSELoss()
# cost_sspcab = mse_loss(input_sspcab, output_sspcab)
