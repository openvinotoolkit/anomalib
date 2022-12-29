"""PyTorch model for CS-Flow implementation."""


# Original Code
# Copyright (c) 2021 marco-rudolph
# https://github.com/marco-rudolph/cs-flow
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from math import exp
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from FrEIA.framework import GraphINN, InputNode, Node, OutputNode
from FrEIA.modules import InvertibleModule
from torch import Tensor, nn
from torchvision.models.efficientnet import EfficientNet_B5_Weights

from anomalib.models.components.feature_extractors import TorchFXFeatureExtractor

from .anomaly_map import AnomalyMapGenerator, AnomalyMapMode


class CrossConvolutions(nn.Module):
    """Cross convolution for the three scales.

    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of output channels in the hidden convolution and the upscaling layers.
        channels_hidden (int, optional): Number of input channels in the hidden convolution layers. Defaults to 512.
        kernel_size (int, optional): Kernel size of the convolution layers. Defaults to 3.
        leaky_slope (float, optional): Slope of the leaky ReLU activation. Defaults to 0.1.
        batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
        use_gamma (bool, optional): Whether to use gamma parameters for the cross convolutions. Defaults to True.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        channels_hidden: int = 512,
        kernel_size: int = 3,
        leaky_slope: float = 0.1,
        batch_norm: bool = False,
        use_gamma: bool = True,
    ):
        super().__init__()

        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        pad_mode = "zeros"
        self.use_gamma = use_gamma
        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.conv_scale0_0 = nn.Conv2d(
            in_channels,
            channels_hidden,
            kernel_size=kernel_size,
            padding=pad,
            bias=not batch_norm,
            padding_mode=pad_mode,
        )

        self.conv_scale1_0 = nn.Conv2d(
            in_channels,
            channels_hidden,
            kernel_size=kernel_size,
            padding=pad,
            bias=not batch_norm,
            padding_mode=pad_mode,
        )
        self.conv_scale2_0 = nn.Conv2d(
            in_channels,
            channels_hidden,
            kernel_size=kernel_size,
            padding=pad,
            bias=not batch_norm,
            padding_mode=pad_mode,
        )
        self.conv_scale0_1 = nn.Conv2d(
            channels_hidden * 1,
            channels,  #
            kernel_size=kernel_size,
            padding=pad,
            bias=not batch_norm,
            padding_mode=pad_mode,
            dilation=1,
        )
        self.conv_scale1_1 = nn.Conv2d(
            channels_hidden * 1,
            channels,  #
            kernel_size=kernel_size,
            padding=pad * 1,
            bias=not batch_norm,
            padding_mode=pad_mode,
            dilation=1,
        )
        self.conv_scale2_1 = nn.Conv2d(
            channels_hidden * 1,
            channels,  #
            kernel_size=kernel_size,
            padding=pad,
            bias=not batch_norm,
            padding_mode=pad_mode,
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.up_conv10 = nn.Conv2d(
            channels_hidden, channels, kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode
        )

        self.up_conv21 = nn.Conv2d(
            channels_hidden, channels, kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode
        )

        self.down_conv01 = nn.Conv2d(
            channels_hidden,
            channels,
            kernel_size=kernel_size,
            padding=pad,
            bias=not batch_norm,
            stride=2,
            padding_mode=pad_mode,
            dilation=1,
        )

        self.down_conv12 = nn.Conv2d(
            channels_hidden,
            channels,
            kernel_size=kernel_size,
            padding=pad,
            bias=not batch_norm,
            stride=2,
            padding_mode=pad_mode,
            dilation=1,
        )

        self.leaky_relu = nn.LeakyReLU(self.leaky_slope)

    def forward(self, scale0, scale1, scale2) -> Tuple[Tensor, Tensor, Tensor]:
        """Applies the cross convolution to the three scales.

        This block is represented in figure 4 of the paper.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Tensors indicating scale and transform parameters as a single tensor for
            each scale. The scale parameters are the first part across channel dimension and the transform parameters
            are the second.
        """
        # Increase the number of channels to hidden channel length via convolutions and apply leaky ReLU.
        out0 = self.conv_scale0_0(scale0)
        out1 = self.conv_scale1_0(scale1)
        out2 = self.conv_scale2_0(scale2)

        lr0 = self.leaky_relu(out0)
        lr1 = self.leaky_relu(out1)
        lr3 = self.leaky_relu(out2)

        # Decrease the number of channels to scale and transform split length.
        out0 = self.conv_scale0_1(lr0)
        out1 = self.conv_scale1_1(lr1)
        out2 = self.conv_scale2_1(lr3)

        # Upsample the smaller scales.
        y1_up = self.up_conv10(self.upsample(lr1))
        y2_up = self.up_conv21(self.upsample(lr3))

        # Downsample the larger scales.
        y0_down = self.down_conv01(lr0)
        y1_down = self.down_conv12(lr1)

        # Do element-wise sum on cross-scale outputs.
        out0 = out0 + y1_up
        out1 = out1 + y0_down + y2_up
        out2 = out2 + y1_down

        if self.use_gamma:
            out0 = out0 * self.gamma0
            out1 = out1 * self.gamma1
            out2 = out2 * self.gamma2
        # even channel split is performed outside this block
        return out0, out1, out2


class ParallelPermute(InvertibleModule):
    """Permutes input vector in a random but fixed way.

    Args:
        dim (List[Tuple[int]]): Dimension of the input vector.
        seed (Optional[float]=None): Seed for the random permutation.
    """

    def __init__(self, dims_in: List[Tuple[int]], seed: Optional[float] = None):
        super().__init__(dims_in)
        self.n_inputs: int = len(dims_in)
        self.in_channels = [dims_in[i][0] for i in range(self.n_inputs)]

        np.random.seed(seed)
        perm, perm_inv = self.get_random_perm(0)
        self.perm = [perm]  # stores the random order of channels
        self.perm_inv = [perm_inv]  # stores the inverse mapping to recover the original order of channels

        for i in range(1, self.n_inputs):
            perm, perm_inv = self.get_random_perm(i)
            self.perm.append(perm)
            self.perm_inv.append(perm_inv)

    def get_random_perm(self, index: int) -> Tuple[Tensor, Tensor]:
        """Returns a random permutation of the channels for each input.

        Args:
            i: index of the input

        Returns:
            Tuple[Tensor, Tensor]: permutation and inverse permutation
        """
        perm = np.random.permutation(self.in_channels[index])
        perm_inv = np.zeros_like(perm)
        for idx, permutation in enumerate(perm):
            perm_inv[permutation] = idx

        perm = torch.LongTensor(perm)
        perm_inv = torch.LongTensor(perm_inv)
        return perm, perm_inv

    # pylint: disable=unused-argument
    def forward(self, input_tensor: List[Tensor], rev=False, jac=True) -> Tuple[List[Tensor], float]:
        """Applies the permutation to the input.

        Args:
            input_tensor: list of input tensors
            rev: if True, applies the inverse permutation
            jac: (unused) if True, computes the log determinant of the Jacobian

        Returns:
            Tuple[Tensor, Tensor]: output tensor and log determinant of the Jacobian
        """
        if not rev:
            return [input_tensor[i][:, self.perm[i]] for i in range(self.n_inputs)], 0.0

        return [input_tensor[i][:, self.perm_inv[i]] for i in range(self.n_inputs)], 0.0

    def output_dims(self, input_dims: List[Tuple[int]]) -> List[Tuple[int]]:
        """Returns the output dimensions of the module."""
        return input_dims


class ParallelGlowCouplingLayer(InvertibleModule):
    """Coupling block that follows the GLOW design but is applied to all the scales in parallel.

    Args:
        dims_in (List[Tuple[int]]): list of dimensions of the input tensors
        subnet_args (Dict): arguments of the subnet
        clamp (float): clamp value for the output of the subnet
    """

    def __init__(self, dims_in: List[Tuple[int]], subnet_args: Dict, clamp: float = 5.0):
        super().__init__(dims_in)
        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])

        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp

        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.cross_convolution1 = CrossConvolutions(self.split_len1, self.split_len2 * 2, **subnet_args)
        self.cross_convolution2 = CrossConvolutions(self.split_len2, self.split_len1 * 2, **subnet_args)

    def exp(self, input_tensor):
        """Exponentiates the input and, optionally, clamps it to avoid numerical issues."""
        if self.clamp > 0:
            return torch.exp(self.log_e(input_tensor))
        return torch.exp(input_tensor)

    def log_e(self, input_tensor):
        """Returns log of input. And optionally clamped to avoid numerical issues."""
        if self.clamp > 0:
            return self.clamp * 0.636 * torch.atan(input_tensor / self.clamp)
        return input_tensor

    # pylint: disable=unused-argument
    def forward(self, input_tensor: List[Tensor], rev=False, jac=True):
        """Applies GLOW coupling for the three scales."""

        # Even channel split. The two splits are used by cross-scale convolution to compute scale and transform
        # parameters.
        x01, x02 = (
            input_tensor[0].narrow(1, 0, self.split_len1),
            input_tensor[0].narrow(1, self.split_len1, self.split_len2),
        )
        x11, x12 = (
            input_tensor[1].narrow(1, 0, self.split_len1),
            input_tensor[1].narrow(1, self.split_len1, self.split_len2),
        )
        x21, x22 = (
            input_tensor[2].narrow(1, 0, self.split_len1),
            input_tensor[2].narrow(1, self.split_len1, self.split_len2),
        )

        if not rev:
            # Outputs of cross convolutions at three scales
            r02, r12, r22 = self.cross_convolution2(x02, x12, x22)

            # Scale and transform parameters are obtained by splitting the output of cross convolutions.
            s02, t02 = r02[:, : self.split_len1], r02[:, self.split_len1 :]
            s12, t12 = r12[:, : self.split_len1], r12[:, self.split_len1 :]
            s22, t22 = r22[:, : self.split_len1], r22[:, self.split_len1 :]

            # apply element wise affine transformation on the first part
            y01 = self.exp(s02) * x01 + t02
            y11 = self.exp(s12) * x11 + t12
            y21 = self.exp(s22) * x21 + t22

            r01, r11, r21 = self.cross_convolution1(y01, y11, y21)

            s01, t01 = r01[:, : self.split_len2], r01[:, self.split_len2 :]
            s11, t11 = r11[:, : self.split_len2], r11[:, self.split_len2 :]
            s21, t21 = r21[:, : self.split_len2], r21[:, self.split_len2 :]

            # apply element wise affine transformation on the second part
            y02 = self.exp(s01) * x02 + t01
            y12 = self.exp(s11) * x12 + t11
            y22 = self.exp(s21) * x22 + t21

        else:  # names of x and y are swapped!
            # Inverse affine transformation at three scales.
            r01, r11, r21 = self.cross_convolution1(x01, x11, x21)

            s01, t01 = r01[:, : self.split_len2], r01[:, self.split_len2 :]
            s11, t11 = r11[:, : self.split_len2], r11[:, self.split_len2 :]
            s21, t21 = r21[:, : self.split_len2], r21[:, self.split_len2 :]

            y02 = (x02 - t01) / self.exp(s01)
            y12 = (x12 - t11) / self.exp(s11)
            y22 = (x22 - t21) / self.exp(s21)

            r02, r12, r22 = self.cross_convolution2(y02, y12, y22)

            s02, t02 = r02[:, : self.split_len2], r01[:, self.split_len2 :]
            s12, t12 = r12[:, : self.split_len2], r11[:, self.split_len2 :]
            s22, t22 = r22[:, : self.split_len2], r21[:, self.split_len2 :]

            y01 = (x01 - t02) / self.exp(s02)
            y11 = (x11 - t12) / self.exp(s12)
            y21 = (x21 - t22) / self.exp(s22)

        # Concatenate the outputs of the three scales to get three transformed outputs that have the same shape as the
        # inputs.
        z_dist0 = torch.cat((y01, y02), 1)
        z_dist1 = torch.cat((y11, y12), 1)
        z_dist2 = torch.cat((y21, y22), 1)

        z_dist0 = torch.clamp(z_dist0, -1e6, 1e6)
        z_dist1 = torch.clamp(z_dist1, -1e6, 1e6)
        z_dist2 = torch.clamp(z_dist2, -1e6, 1e6)

        jac0 = torch.sum(self.log_e(s01), dim=(1, 2, 3)) + torch.sum(self.log_e(s02), dim=(1, 2, 3))
        jac1 = torch.sum(self.log_e(s11), dim=(1, 2, 3)) + torch.sum(self.log_e(s12), dim=(1, 2, 3))
        jac2 = torch.sum(self.log_e(s21), dim=(1, 2, 3)) + torch.sum(self.log_e(s22), dim=(1, 2, 3))

        # Since Jacobians are only used for computing loss and summed in the loss, the idea is to sum them here
        return [z_dist0, z_dist1, z_dist2], torch.stack([jac0, jac1, jac2], dim=1).sum()

    def output_dims(self, input_dims: List[Tuple[int]]):
        """Output dimensions of the module."""
        return input_dims


class CrossScaleFlow(nn.Module):
    """Cross scale coupling layer.

    Args:
        input_dims (Tuple[int, int, int]): Input dimensions of the module.
        n_coupling_blocks (int): Number of coupling blocks.
        clamp (float): Clamp value for the inputs.
        corss_conv_hidden_channels (int): Number of hidden channels in the cross convolution.
    """

    def __init__(
        self, input_dims: Tuple[int, int, int], n_coupling_blocks: int, clamp: float, cross_conv_hidden_channels: int
    ):
        super().__init__()
        self.input_dims = input_dims
        self.n_coupling_blocks = n_coupling_blocks
        self.kernel_sizes = [3] * (n_coupling_blocks - 1) + [5]
        self.clamp = clamp
        self.cross_conv_hidden_channels = cross_conv_hidden_channels
        self.graph = self._create_graph()

    def _create_graph(self):
        nodes = []
        # 304 is the number of features extracted from EfficientNet-B5 feature extractor
        nodes.append(InputNode(304, (self.input_dims[1] // 32), (self.input_dims[2] // 32), name="input"))
        nodes.append(InputNode(304, (self.input_dims[1] // 64), (self.input_dims[2] // 64), name="input2"))
        nodes.append(InputNode(304, (self.input_dims[1] // 128), (self.input_dims[2] // 128), name="input3"))

        for coupling_block in range(self.n_coupling_blocks):
            if coupling_block == 0:
                node_to_permute = [nodes[-3].out0, nodes[-2].out0, nodes[-1].out0]
            else:
                node_to_permute = [nodes[-1].out0, nodes[-1].out1, nodes[-1].out2]

            nodes.append(
                Node(node_to_permute, ParallelPermute, {"seed": coupling_block}, name=f"permute_{coupling_block}")
            )
            nodes.append(
                Node(
                    [nodes[-1].out0, nodes[-1].out1, nodes[-1].out2],
                    ParallelGlowCouplingLayer,
                    {
                        "clamp": self.clamp,
                        "subnet_args": {
                            "channels_hidden": self.cross_conv_hidden_channels,
                            "kernel_size": self.kernel_sizes[coupling_block],
                        },
                    },
                    name=f"fc1_{coupling_block}",
                )
            )

        nodes.append(OutputNode([nodes[-1].out0], name="output_end0"))
        nodes.append(OutputNode([nodes[-2].out1], name="output_end1"))
        nodes.append(OutputNode([nodes[-3].out2], name="output_end2"))
        return GraphINN(nodes)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            inputs (Tensor): Input tensor.

        Returns:
            Tuple[Tensor, Tensor]: Output tensor and log determinant of Jacobian.
        """
        return self.graph(inputs)


class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extractor.

    Uses 36th layer of EfficientNet-B5 to extract features.

    Args:
        n_scales (int): Number of scales for input image.
        input_size (Tuple[int, int]): Size of input image.
    """

    def __init__(self, n_scales: int, input_size: Tuple[int, int]):
        super().__init__()

        self.n_scales = n_scales
        self.input_size = input_size
        self.feature_extractor = TorchFXFeatureExtractor(
            backbone="efficientnet_b5", weights=EfficientNet_B5_Weights.DEFAULT, return_nodes=["features.6.8"]
        )

    def forward(self, input_tensor: Tensor) -> List[Tensor]:
        """Extracts features at three scales.

        Args:
            input_tensor (Tensor): Input images.

        Returns:
            List[Tensor]: List of tensors containing features at three scales.
        """
        output = []
        for scale in range(self.n_scales):
            feat_s = (
                F.interpolate(
                    input_tensor, size=(self.input_size[0] // (2**scale), self.input_size[1] // (2**scale))
                )
                if scale > 0
                else input_tensor
            )
            feat_s = self.feature_extractor(feat_s)["features.6.8"]

            output.append(feat_s)
        return output


class CsFlowModel(nn.Module):
    """CS Flow Module.

    Args:
        input_size (Tuple[int, int]): Input image size.
        cross_conv_hidden_channels (int): Number of hidden channels in the cross convolution.
        n_coupling_blocks (int): Number of coupling blocks.
        clamp (float): Clamp value for the coupling blocks.
        num_channels (int): Number of channels in the input image.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        cross_conv_hidden_channels: int,
        n_coupling_blocks: int = 4,
        clamp: int = 3,
        num_channels: int = 3,
    ):

        super().__init__()
        self.input_dims = (num_channels, *input_size)
        self.clamp = clamp
        self.cross_conv_hidden_channels = cross_conv_hidden_channels
        self.feature_extractor = MultiScaleFeatureExtractor(n_scales=3, input_size=input_size)
        self.graph = CrossScaleFlow(
            input_dims=self.input_dims,
            n_coupling_blocks=n_coupling_blocks,
            clamp=clamp,
            cross_conv_hidden_channels=cross_conv_hidden_channels,
        )
        self.anomaly_map_generator = AnomalyMapGenerator(input_dims=self.input_dims, mode=AnomalyMapMode.ALL)

    def forward(self, images) -> Tuple[Tensor, Tensor]:
        """Forward method of the model.

        Args:
            images (Tensor): Input images.

        Returns:
            Tuple[Tensor, Tensor]: During training: Tuple containing the z_distribution for three scales and the sum
                of log determinant of the Jacobian. During evaluation: Tuple containing anomaly maps and anomaly scores
        """
        features = self.feature_extractor(images)
        if self.training:
            output = self.graph(features)
        else:
            z_dist, _ = self.graph(features)  # Ignore Jacobians
            anomaly_scores = self._compute_anomaly_scores(z_dist)
            anomaly_maps = self.anomaly_map_generator(z_dist)
            output = anomaly_maps, anomaly_scores
        return output

    def _compute_anomaly_scores(self, z_dists: Tensor) -> Tensor:
        """Get anomaly scores from the latent distribution.

        Args:
            z_dist (Tensor): Latent distribution.

        Returns:
            Tensor: Anomaly scores.
        """
        # z_dist is a 3 length list of tensors with shape b x 304 x fx x fy
        flat_maps: List[Tensor] = []
        for z_dist in z_dists:
            flat_maps.append(z_dist.reshape(z_dist.shape[0], -1))
        flat_maps_tensor = torch.cat(flat_maps, dim=1)
        anomaly_scores = torch.mean(flat_maps_tensor**2 / 2, dim=1)
        return anomaly_scores
