"""U-Flow PyTorch Implementation.

This module provides the PyTorch implementation of the U-Flow model for anomaly detection.
U-Flow combines normalizing flows with a U-Net style architecture to learn the distribution
of normal images and detect anomalies.

The model consists of several key components:
    - Feature extraction using a pre-trained backbone
    - Normalizing flow blocks arranged in a U-Net structure
    - Anomaly map generation for localization

The implementation includes classes for:
    - Affine coupling subnet construction
    - Main U-Flow model architecture
    - Anomaly map generation
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from FrEIA import framework as ff
from FrEIA import modules as fm
from torch import nn

from anomalib.data import InferenceBatch
from anomalib.models.components.flow import AllInOneBlock

from .anomaly_map import AnomalyMapGenerator
from .feature_extraction import get_feature_extractor


class AffineCouplingSubnet:
    """Class for building the Affine Coupling subnet.

    This class creates a subnet used within the affine coupling layers of the normalizing
    flow. The subnet is passed as an argument to the ``AllInOneBlock`` module and
    determines how features are transformed within the coupling layer.

    Args:
        kernel_size (int): Size of convolutional kernels used in subnet layers.
        subnet_channels_ratio (float): Ratio determining the number of intermediate
            channels in the subnet relative to input channels.

    Example:
        >>> subnet = AffineCouplingSubnet(kernel_size=3, subnet_channels_ratio=1.0)
        >>> layer = subnet(in_channels=64, out_channels=128)
        >>> layer
        Sequential(
          (0): Conv2d(64, 64, kernel_size=(3, 3), padding=same)
          (1): ReLU()
          (2): Conv2d(64, 128, kernel_size=(3, 3), padding=same)
        )

    See Also:
        - :class:`AllInOneBlock`: Flow block using this subnet
        - :class:`UflowModel`: Main model incorporating these subnets
    """

    def __init__(self, kernel_size: int, subnet_channels_ratio: float) -> None:
        self.kernel_size = kernel_size
        self.subnet_channels_ratio = subnet_channels_ratio

    def __call__(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Create and return the affine coupling subnet.

        The subnet consists of two convolutional layers with a ReLU activation in
        between. The intermediate channel dimension is determined by
        ``subnet_channels_ratio``.

        Args:
            in_channels (int): Number of input channels to the subnet.
            out_channels (int): Number of output channels from the subnet.

        Returns:
            nn.Sequential: Sequential container of the subnet layers including:
                - Conv2d layer mapping input to intermediate channels
                - ReLU activation
                - Conv2d layer mapping intermediate to output channels
        """
        mid_channels = int(in_channels * self.subnet_channels_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, self.kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, self.kernel_size, padding="same"),
        )


class UflowModel(nn.Module):
    """PyTorch implementation of the U-Flow model architecture.

    This class implements the U-Flow model for anomaly detection.
    The model consists of:

    - A U-shaped normalizing flow architecture for density estimation
    - Multi-scale feature extraction using pre-trained backbones
    - Unsupervised threshold estimation based on the learned density
    - Anomaly detection by comparing likelihoods to the threshold

    Args:
        input_size (tuple[int, int]): Input image dimensions as ``(height, width)``.
            Defaults to ``(448, 448)``.
        flow_steps (int): Number of normalizing flow steps in each flow stage.
            Defaults to ``4``.
        backbone (str): Name of the backbone feature extractor. Must be one of
            ``["mcait", "resnet18", "wide_resnet50_2"]``. Defaults to ``"mcait"``.
        affine_clamp (float): Clamping value for affine coupling layers. Defaults
            to ``2.0``.
        affine_subnet_channels_ratio (float): Channel ratio for affine coupling
            subnet. Defaults to ``1.0``.
        permute_soft (bool): Whether to use soft permutation. Defaults to
            ``False``.

    Example:
        >>> import torch
        >>> from anomalib.models.image.uflow.torch_model import UflowModel
        >>> model = UflowModel(
        ...     input_size=(256, 256),
        ...     backbone="resnet18"
        ... )
        >>> image = torch.randn(1, 3, 256, 256)
        >>> output = model(image)  # Returns anomaly map during inference

    See Also:
        - :class:`Uflow`: Lightning implementation using this model
        - :class:`UFlowLoss`: Loss function for training
        - :class:`AnomalyMapGenerator`: Anomaly map generation from features
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
        """Build the U-shaped normalizing flow architecture.

        The flow is built in a U-shaped structure, processing features from coarse
        to fine scales:

        1. Start with input nodes matching feature extractor outputs
        2. For each scale (coarse to fine):
            - Pass through flow stage (sequence of coupling layers)
            - Split output into two parts
            - Send one part to output
            - Upsample other part and concatenate with next scale
        3. Build final flow graph combining all nodes

        Args:
            flow_steps (int): Number of coupling layers in each flow stage.

        Returns:
            ff.GraphINN: Constructed normalizing flow graph.

        See Also:
            - :meth:`build_flow_stage`: Builds individual flow stages
            - :class:`AllInOneBlock`: Individual coupling layer blocks
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
        """Build a single flow stage consisting of multiple coupling layers.

        Each flow stage is a sequence of ``flow_steps`` Glow-style coupling blocks
        (``AllInOneBlock``). The blocks alternate between 3x3 and 1x1 convolutions
        in their coupling subnets.

        Args:
            in_node (ff.Node): Input node to the flow stage.
            flow_steps (int): Number of coupling layers to use.
            condition_node (ff.Node, optional): Optional conditioning node.
                Defaults to ``None``.

        Returns:
            list[ff.Node]: List of constructed coupling layer nodes.

        See Also:
            - :class:`AllInOneBlock`: Individual coupling layer implementation
            - :class:`AffineCouplingSubnet`: Subnet used in coupling layers
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

    def forward(self, image: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """Process input image through the model.

        During training, returns latent variables and log-Jacobian determinant.
        During inference, returns anomaly scores and anomaly map.

        Args:
            image (torch.Tensor): Input image tensor of shape
                ``(batch_size, channels, height, width)``.

        Returns:
            torch.Tensor | InferenceBatch: During training, returns tuple of
                ``(latent_vars, log_jacobian)``. During inference, returns
                ``InferenceBatch`` with anomaly scores and map.
        """
        features = self.feature_extractor(image)
        z, ljd = self.encode(features)

        if self.training:
            return z, ljd

        anomaly_map = self.anomaly_map_generator(z)
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

    def encode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input features to latent space using normalizing flow.

        Args:
            features (torch.Tensor): Input features from feature extractor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - Latent variables from flow transformation
                - Log-Jacobian determinant of the transformation
        """
        z, ljd = self.flow(features, rev=False)
        if len(self.feature_extractor.scales) == 1:
            z = [z]
        return z, ljd
