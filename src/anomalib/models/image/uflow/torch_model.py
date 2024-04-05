"""U-Flow torch model."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from FrEIA import framework as ff
from FrEIA import modules as fm
from torch import nn

from anomalib.models.components.flow import AllInOneBlock

from .anomaly_map import AnomalyMapGenerator
from .feature_extraction import get_feature_extractor


class AffineCouplingSubnet:
    """Class for building the Affine Coupling subnet.

    It is passed as an argument to the `AllInOneBlock` module.

    Args:
        kernel_size (int): Kernel size.
        subnet_channels_ratio (float): Subnet channels ratio.
    """

    def __init__(self, kernel_size: int, subnet_channels_ratio: float) -> None:
        self.kernel_size = kernel_size
        self.subnet_channels_ratio = subnet_channels_ratio

    def __call__(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Return AffineCouplingSubnet network.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.

        Returns:
            nn.Sequential: Affine Coupling subnet.
        """
        mid_channels = int(in_channels * self.subnet_channels_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, self.kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, self.kernel_size, padding="same"),
        )


class UflowModel(nn.Module):
    """U-Flow model.

    Args:
        input_size (tuple[int, int]): Input image size.
        flow_steps (int): Number of flow steps.
        backbone (str): Backbone name.
        affine_clamp (float): Affine clamp.
        affine_subnet_channels_ratio (float): Affine subnet channels ratio.
        permute_soft (bool): Whether to use soft permutation.
    """

    def __init__(
        self,
        input_size: tuple[int, int] = (448, 448),
        flow_steps: int = 4,
        backbone: str = "mcait",
        affine_clamp: float = 2.0,
        affine_subnet_channels_ratio: float = 1.0,
        permute_soft: bool = False,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.affine_clamp = affine_clamp
        self.affine_subnet_channels_ratio = affine_subnet_channels_ratio
        self.permute_soft = permute_soft

        self.feature_extractor = get_feature_extractor(backbone, input_size)
        self.flow = self.build_flow(flow_steps)
        self.anomaly_map_generator = AnomalyMapGenerator(input_size)

    def build_flow(self, flow_steps: int) -> ff.GraphINN:
        """Build the flow model.

        First we start with the input nodes, which have to match the feature extractor output.
        Then, we build the U-Shaped flow. Starting from the bottom (the coarsest scale), the flow is built as follows:
            1. Pass the input through a Flow Stage (`build_flow_stage`).
            2. Split the output of the flow stage into two parts, one that goes directly to the output,
            3. and the other is up-sampled, and will be concatenated with the output of the next flow stage (next scale)
            4. Repeat steps 1-3 for the next scale.
        Finally, we build the Flow graph using the input nodes, the flow stages, and the output nodes.

        Args:
            flow_steps (int): Number of flow steps.

        Returns:
            ff.GraphINN: Flow model.
        """
        input_nodes = []
        for channel, s_factor in zip(
            self.feature_extractor.channels,
            self.feature_extractor.scale_factors,
            strict=True,
        ):
            input_nodes.append(
                ff.InputNode(
                    channel,
                    self.input_size[0] // s_factor,
                    self.input_size[1] // s_factor,
                    name=f"cond_{channel}",
                ),
            )

        nodes, output_nodes = [], []
        last_node = input_nodes[-1]
        for i in reversed(range(1, len(input_nodes))):
            flows = self.build_flow_stage(last_node, flow_steps)
            volume_size = flows[-1].output_dims[0][0]
            split = ff.Node(
                flows[-1],
                fm.Split,
                {"section_sizes": (volume_size // 8 * 4, volume_size - volume_size // 8 * 4), "dim": 0},
                name=f"split_{i + 1}",
            )
            output = ff.OutputNode(split.out1, name=f"output_scale_{i + 1}")
            up = ff.Node(split.out0, fm.IRevNetUpsampling, {}, name=f"up_{i + 1}")
            last_node = ff.Node([input_nodes[i - 1].out0, up.out0], fm.Concat, {"dim": 0}, name=f"cat_{i}")

            output_nodes.append(output)
            nodes.extend([*flows, split, up, last_node])

        flows = self.build_flow_stage(last_node, flow_steps)
        output = ff.OutputNode(flows[-1], name="output_scale_1")

        output_nodes.append(output)
        nodes.extend(flows)

        return ff.GraphINN(input_nodes + nodes + output_nodes[::-1])

    def build_flow_stage(self, in_node: ff.Node, flow_steps: int, condition_node: ff.Node = None) -> list[ff.Node]:
        """Build a flow stage, which is a sequence of flow steps.

        Each flow stage is essentially a sequence of `flow_steps` Glow blocks (`AllInOneBlock`).

        Args:
            in_node (ff.Node): Input node.
            flow_steps (int): Number of flow steps.
            condition_node (ff.Node): Condition node.

        Returns:
            List[ff.Node]: List of flow steps.
        """
        flow_size = in_node.output_dims[0][-1]
        nodes = []
        for step in range(flow_steps):
            nodes.append(
                ff.Node(
                    in_node,
                    AllInOneBlock,
                    module_args={
                        "subnet_constructor": AffineCouplingSubnet(
                            3 if step % 2 == 0 else 1,
                            self.affine_subnet_channels_ratio,
                        ),
                        "affine_clamping": self.affine_clamp,
                        "permute_soft": self.permute_soft,
                    },
                    conditions=condition_node,
                    name=f"flow{flow_size}_step{step}",
                ),
            )
            in_node = nodes[-1]
        return nodes

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return anomaly map."""
        features = self.feature_extractor(image)
        z, ljd = self.encode(features)

        if self.training:
            return z, ljd
        return self.anomaly_map_generator(z)

    def encode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return"""
        z, ljd = self.flow(features, rev=False)
        if len(self.feature_extractor.scales) == 1:
            z = [z]
        return z, ljd
