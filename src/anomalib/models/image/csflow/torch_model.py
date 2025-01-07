"""PyTorch model for CS-Flow implementation.

This module contains the PyTorch implementation of CS-Flow model for anomaly detection.
The model uses cross-scale coupling layers to learn the distribution of normal images
and detect anomalies based on the likelihood of test images under this distribution.

The implementation is based on the paper:
    CS-Flow: Learning Cross-Scale Semantic Flow for Unsupervised Anomaly Detection
    Marco Rudolph, Tom Wehrbein, Bodo Rosenhahn, Bastian Wandt
    https://arxiv.org/abs/2110.02855
"""

# Original Code
# Copyright (c) 2021 marco-rudolph
# https://github.com/marco-rudolph/cs-flow
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from math import exp

import numpy as np
import torch
from FrEIA.framework import GraphINN, InputNode, Node, OutputNode
from FrEIA.modules import InvertibleModule
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torchvision.models.efficientnet import EfficientNet_B5_Weights

from anomalib.data import InferenceBatch
from anomalib.models.components.feature_extractors import TorchFXFeatureExtractor

from .anomaly_map import AnomalyMapGenerator, AnomalyMapMode


class CrossConvolutions(nn.Module):
    """Cross convolution module for processing features at three scales.

    This module applies convolutions across three different scales of features,
    with connections between scales via up/downsampling operations.

    Args:
        in_channels (int): Number of input channels.
        channels (int): Number of output channels in convolution layers.
        channels_hidden (int, optional): Number of channels in hidden layers.
            Defaults to ``512``.
        kernel_size (int, optional): Size of convolution kernels.
            Defaults to ``3``.
        leaky_slope (float, optional): Negative slope for leaky ReLU.
            Defaults to ``0.1``.
        batch_norm (bool, optional): Whether to use batch normalization.
            Defaults to ``False``.
        use_gamma (bool, optional): Whether to use learnable gamma parameters.
            Defaults to ``True``.

    Example:
        >>> cross_conv = CrossConvolutions(64, 128)
        >>> scale0 = torch.randn(1, 64, 32, 32)
        >>> scale1 = torch.randn(1, 64, 16, 16)
        >>> scale2 = torch.randn(1, 64, 8, 8)
        >>> out0, out1, out2 = cross_conv(scale0, scale1, scale2)
        >>> out0.shape, out1.shape, out2.shape
        (torch.Size([1, 128, 32, 32]), torch.Size([1, 128, 16, 16]),
         torch.Size([1, 128, 8, 8]))
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
    ) -> None:
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
            channels,
            kernel_size=kernel_size,
            padding=pad,
            bias=not batch_norm,
            padding_mode=pad_mode,
            dilation=1,
        )
        self.conv_scale1_1 = nn.Conv2d(
            channels_hidden * 1,
            channels,
            kernel_size=kernel_size,
            padding=pad * 1,
            bias=not batch_norm,
            padding_mode=pad_mode,
            dilation=1,
        )
        self.conv_scale2_1 = nn.Conv2d(
            channels_hidden * 1,
            channels,
            kernel_size=kernel_size,
            padding=pad,
            bias=not batch_norm,
            padding_mode=pad_mode,
        )

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.up_conv10 = nn.Conv2d(
            channels_hidden,
            channels,
            kernel_size=kernel_size,
            padding=pad,
            bias=True,
            padding_mode=pad_mode,
        )

        self.up_conv21 = nn.Conv2d(
            channels_hidden,
            channels,
            kernel_size=kernel_size,
            padding=pad,
            bias=True,
            padding_mode=pad_mode,
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

    def forward(self, scale0: int, scale1: int, scale2: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply cross-scale convolutions to input features.

        Processes features at three scales with cross-connections between scales via
        up/downsampling operations. This implements the architecture shown in Figure 4
        of the CS-Flow paper.

        Args:
            scale0 (torch.Tensor): Features at original scale.
            scale1 (torch.Tensor): Features at 1/2 scale.
            scale2 (torch.Tensor): Features at 1/4 scale.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Processed features at three
                scales. Each tensor contains scale and transform parameters concatenated
                along the channel dimension.
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
    """Permutes input vectors in a random but fixed way.

    This module applies a fixed random permutation to the channels of each input
    tensor. The permutation is deterministic for a given seed.

    Args:
        dims_in (list[tuple[int]]): List of input tensor dimensions.
        seed (int | None, optional): Random seed for permutation.
            Defaults to ``None``.

    Example:
        >>> permute = ParallelPermute([(3, 32, 32), (3, 16, 16)], seed=42)
        >>> x1 = torch.randn(1, 3, 32, 32)
        >>> x2 = torch.randn(1, 3, 16, 16)
        >>> y1, y2 = permute([x1, x2])[0]
        >>> y1.shape, y2.shape
        (torch.Size([1, 3, 32, 32]), torch.Size([1, 3, 16, 16]))
    """

    def __init__(self, dims_in: list[tuple[int]], seed: int | None = None) -> None:
        super().__init__(dims_in)
        self.n_inputs: int = len(dims_in)
        self.in_channels = [dims_in[i][0] for i in range(self.n_inputs)]
        self.seed = seed

        perm, perm_inv = self.get_random_perm(0)
        self.perm = [perm]  # stores the random order of channels
        self.perm_inv = [perm_inv]  # stores the inverse mapping to recover the original order of channels

        for i in range(1, self.n_inputs):
            perm, perm_inv = self.get_random_perm(i)
            self.perm.append(perm)
            self.perm_inv.append(perm_inv)

    def get_random_perm(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate random permutation and its inverse for given input index.

        Args:
            index (int): Index of input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Permutation and inverse permutation tensors.
        """
        perm = np.random.default_rng(self.seed).permutation(self.in_channels[index])
        perm_inv = np.zeros_like(perm)
        for idx, permutation in enumerate(perm):
            perm_inv[permutation] = idx

        perm = torch.LongTensor(perm)
        perm_inv = torch.LongTensor(perm_inv)
        return perm, perm_inv

    # pylint: disable=unused-argument
    def forward(
        self,
        input_tensor: list[torch.Tensor],
        rev: bool = False,
        jac: bool = True,
    ) -> tuple[list[torch.Tensor], float]:
        """Apply permutation or inverse permutation to inputs.

        Args:
            input_tensor (list[torch.Tensor]): List of input tensors.
            rev (bool, optional): If ``True``, applies inverse permutation.
                Defaults to ``False``.
            jac (bool, optional): Unused. Required for interface compatibility.
                Defaults to ``True``.

        Returns:
            tuple[list[torch.Tensor], float]: Permuted tensors and log determinant (0).
        """
        del jac  # Unused argument.

        if not rev:
            return [input_tensor[i][:, self.perm[i]] for i in range(self.n_inputs)], 0.0

        return [input_tensor[i][:, self.perm_inv[i]] for i in range(self.n_inputs)], 0.0

    @staticmethod
    def output_dims(input_dims: list[tuple[int]]) -> list[tuple[int]]:
        """Return output dimensions of the module.

        Args:
            input_dims (list[tuple[int]]): List of input dimensions.

        Returns:
            list[tuple[int]]: List of output dimensions (same as input).
        """
        return input_dims


class ParallelGlowCouplingLayer(InvertibleModule):
    """Coupling block following GLOW design applied to multiple scales in parallel.

    This module implements an invertible coupling layer that processes multiple scales
    simultaneously, following the GLOW architecture design.

    Args:
        dims_in (list[tuple[int]]): List of input tensor dimensions.
        subnet_args (dict): Arguments for subnet construction.
        clamp (float, optional): Clamp value for outputs.
            Defaults to ``5.0``.

    Example:
        >>> coupling = ParallelGlowCouplingLayer(
        ...     [(6, 32, 32), (6, 16, 16)],
        ...     {"channels_hidden": 64}
        ... )
        >>> x1 = torch.randn(1, 6, 32, 32)
        >>> x2 = torch.randn(1, 6, 16, 16)
        >>> y1, y2 = coupling([x1, x2])[0]
        >>> y1.shape, y2.shape
        (torch.Size([1, 6, 32, 32]), torch.Size([1, 6, 16, 16]))
    """

    def __init__(self, dims_in: list[tuple[int]], subnet_args: dict, clamp: float = 5.0) -> None:
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

    def exp(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Exponentiates input with optional clamping.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Exponentiated tensor, optionally clamped.
        """
        if self.clamp > 0:
            return torch.exp(self.log_e(input_tensor))
        return torch.exp(input_tensor)

    def log_e(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Compute log with optional clamping.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Log of input, optionally clamped.
        """
        if self.clamp > 0:
            return self.clamp * 0.636 * torch.atan(input_tensor / self.clamp)
        return input_tensor

    def forward(
        self,
        input_tensor: list[torch.Tensor],
        rev: bool = False,
        jac: bool = True,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Apply GLOW coupling transformation to inputs at multiple scales.

        Args:
            input_tensor (list[torch.Tensor]): List of input tensors at different scales.
            rev (bool, optional): If ``True``, applies inverse transformation.
                Defaults to ``False``.
            jac (bool, optional): Unused. Required for interface compatibility.
                Defaults to ``True``.

        Returns:
            tuple[list[torch.Tensor], torch.Tensor]: Transformed tensors and log
                determinant of Jacobian.
        """
        del jac  # Unused argument.

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

    @staticmethod
    def output_dims(input_dims: list[tuple[int]]) -> list[tuple[int]]:
        """Return output dimensions of the module.

        Args:
            input_dims (list[tuple[int]]): List of input dimensions.

        Returns:
            list[tuple[int]]: List of output dimensions (same as input).
        """
        return input_dims


class CrossScaleFlow(nn.Module):
    """Cross scale coupling layer.

    This module implements the cross-scale flow architecture that couples features
    across multiple scales.

    Args:
        input_dims (tuple[int, int, int]): Input dimensions (C, H, W).
        n_coupling_blocks (int): Number of coupling blocks to use.
        clamp (float): Clamping value for coupling layers.
        cross_conv_hidden_channels (int): Hidden channels in cross convolutions.

    Example:
        >>> flow = CrossScaleFlow((3, 256, 256), 4, 3.0, 64)
        >>> x = [
        ...     torch.randn(1, 304, 8, 8),
        ...     torch.randn(1, 304, 4, 4),
        ...     torch.randn(1, 304, 2, 2)
        ... ]
        >>> z, jac = flow(x)
        >>> [zi.shape for zi in z]
        [torch.Size([1, 304, 8, 8]), torch.Size([1, 304, 4, 4]),
         torch.Size([1, 304, 2, 2])]
    """

    def __init__(
        self,
        input_dims: tuple[int, int, int],
        n_coupling_blocks: int,
        clamp: float,
        cross_conv_hidden_channels: int,
    ) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.n_coupling_blocks = n_coupling_blocks
        self.kernel_sizes = [3] * (n_coupling_blocks - 1) + [5]
        self.clamp = clamp
        self.cross_conv_hidden_channels = cross_conv_hidden_channels
        self.graph = self._create_graph()

    def _create_graph(self) -> GraphINN:
        """Create the invertible neural network graph.

        Returns:
            GraphINN: Constructed invertible neural network.
        """
        nodes: list[Node] = []
        # 304 is the number of features extracted from EfficientNet-B5 feature extractor
        input_nodes = [
            InputNode(304, (self.input_dims[1] // 32), (self.input_dims[2] // 32), name="input"),
            InputNode(304, (self.input_dims[1] // 64), (self.input_dims[2] // 64), name="input2"),
            InputNode(304, (self.input_dims[1] // 128), (self.input_dims[2] // 128), name="input3"),
        ]
        nodes.extend(input_nodes)

        for coupling_block in range(self.n_coupling_blocks):
            if coupling_block == 0:
                node_to_permute = [nodes[-3].out0, nodes[-2].out0, nodes[-1].out0]
            else:
                node_to_permute = [nodes[-1].out0, nodes[-1].out1, nodes[-1].out2]

            permute_node = Node(
                inputs=node_to_permute,
                module_type=ParallelPermute,
                module_args={"seed": coupling_block},
                name=f"permute_{coupling_block}",
            )
            nodes.extend([permute_node])
            coupling_layer_node = Node(
                inputs=[nodes[-1].out0, nodes[-1].out1, nodes[-1].out2],
                module_type=ParallelGlowCouplingLayer,
                module_args={
                    "clamp": self.clamp,
                    "subnet_args": {
                        "channels_hidden": self.cross_conv_hidden_channels,
                        "kernel_size": self.kernel_sizes[coupling_block],
                    },
                },
                name=f"fc1_{coupling_block}",
            )
            nodes.extend([coupling_layer_node])

        output_nodes = [
            OutputNode([nodes[-1].out0], name="output_end0"),
            OutputNode([nodes[-1].out1], name="output_end1"),
            OutputNode([nodes[-1].out2], name="output_end2"),
        ]
        nodes.extend(output_nodes)
        return GraphINN(nodes)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the flow model.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor and log determinant
                of Jacobian.
        """
        return self.graph(inputs)


class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extractor using EfficientNet-B5.

    This module extracts features at multiple scales using the 36th layer of
    EfficientNet-B5.

    Args:
        n_scales (int): Number of scales to extract features at.
        input_size (tuple[int, int]): Input image size (H, W).

    Example:
        >>> extractor = MultiScaleFeatureExtractor(3, (256, 256))
        >>> x = torch.randn(1, 3, 256, 256)
        >>> features = extractor(x)
        >>> [f.shape for f in features]
        [torch.Size([1, 304, 8, 8]), torch.Size([1, 304, 4, 4]),
         torch.Size([1, 304, 2, 2])]
    """

    def __init__(self, n_scales: int, input_size: tuple[int, int]) -> None:
        super().__init__()

        self.n_scales = n_scales
        self.input_size = input_size
        self.feature_extractor = TorchFXFeatureExtractor(
            backbone="efficientnet_b5",
            weights=EfficientNet_B5_Weights.DEFAULT,
            return_nodes=["features.6.8"],
        )

    def forward(self, input_tensor: torch.Tensor) -> list[torch.Tensor]:
        """Extract features at multiple scales.

        Args:
            input_tensor (torch.Tensor): Input images.

        Returns:
            list[torch.Tensor]: List of feature tensors at different scales.
        """
        output = []
        for scale in range(self.n_scales):
            feat_s = (
                F.interpolate(
                    input_tensor,
                    size=(self.input_size[0] // (2**scale), self.input_size[1] // (2**scale)),
                )
                if scale > 0
                else input_tensor
            )
            feat_s = self.feature_extractor(feat_s)["features.6.8"]

            output.append(feat_s)
        return output


class CsFlowModel(nn.Module):
    """CS-Flow model for anomaly detection.

    This module implements the complete CS-Flow model that learns the distribution
    of normal images using cross-scale coupling layers.

    Args:
        input_size (tuple[int, int]): Input image size (H, W).
        cross_conv_hidden_channels (int): Hidden channels in cross convolutions.
        n_coupling_blocks (int, optional): Number of coupling blocks.
            Defaults to ``4``.
        clamp (int, optional): Clamping value for coupling layers.
            Defaults to ``3``.
        num_channels (int, optional): Number of input image channels.
            Defaults to ``3``.

    Example:
        >>> model = CsFlowModel((256, 256), 64)
        >>> x = torch.randn(1, 3, 256, 256)
        >>> output = model(x)
        >>> isinstance(output, InferenceBatch)
        True
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        cross_conv_hidden_channels: int,
        n_coupling_blocks: int = 4,
        clamp: int = 3,
        num_channels: int = 3,
    ) -> None:
        super().__init__()
        self.input_dims = (num_channels, *input_size)
        self.clamp = clamp
        self.cross_conv_hidden_channels = cross_conv_hidden_channels
        self.feature_extractor = MultiScaleFeatureExtractor(n_scales=3, input_size=input_size).eval()
        self.graph = CrossScaleFlow(
            input_dims=self.input_dims,
            n_coupling_blocks=n_coupling_blocks,
            clamp=clamp,
            cross_conv_hidden_channels=cross_conv_hidden_channels,
        )
        self.anomaly_map_generator = AnomalyMapGenerator(input_dims=self.input_dims, mode=AnomalyMapMode.ALL)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | InferenceBatch:
        """Forward method of the model.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: During training: tuple containing the z_distribution for three scales
                and the sum of log determinant of the Jacobian. During evaluation: tuple containing anomaly maps
                and anomaly scores
        """
        features = self.feature_extractor(images)
        if self.training:
            return self.graph(features)

        z_dist, _ = self.graph(features)  # Ignore Jacobians
        anomaly_scores = self._compute_anomaly_scores(z_dist)
        anomaly_maps = self.anomaly_map_generator(z_dist)
        return InferenceBatch(pred_score=anomaly_scores, anomaly_map=anomaly_maps)

    @staticmethod
    def _compute_anomaly_scores(z_dists: torch.Tensor) -> torch.Tensor:
        """Get anomaly scores from the latent distribution.

        Args:
            z_dists (torch.Tensor): Latent distribution.

        Returns:
            Tensor: Anomaly scores.
        """
        # z_dist is a 3 length list of tensors with shape b x 304 x fx x fy
        flat_maps = [z_dist.reshape(z_dist.shape[0], -1) for z_dist in z_dists]
        flat_maps_tensor = torch.cat(flat_maps, dim=1)
        return torch.mean(flat_maps_tensor**2 / 2, dim=1)
