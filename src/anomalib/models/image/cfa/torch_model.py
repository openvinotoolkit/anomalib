"""Torch Implementation of the CFA Model.

CFA: Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly
Localization.

This module provides the PyTorch implementation of the CFA model for anomaly
detection and localization. The model learns discriminative features by adapting
them to coupled hyperspheres in the feature space.

The model consists of:
    - A backbone CNN feature extractor
    - A descriptor network that generates target-oriented features
    - A memory bank that stores prototypical normal features
    - An anomaly map generator for localization

Paper: https://arxiv.org/abs/2206.04325

Example:
    >>> import torch
    >>> from anomalib.models.image.cfa.torch_model import CfaModel
    >>> # Initialize model
    >>> model = CfaModel(
    ...     backbone="resnet18",
    ...     gamma_c=1,
    ...     gamma_d=1,
    ...     num_nearest_neighbors=3,
    ...     num_hard_negative_features=3,
    ...     radius=0.5
    ... )
    >>> # Forward pass
    >>> x = torch.randn(32, 3, 256, 256)
    >>> predictions = model(x)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
import torchvision
from einops import rearrange
from sklearn.cluster import KMeans
from torch import nn
from torch.fx.graph_module import GraphModule
from torch.nn import functional as F  # noqa: N812
from torch.nn.common_types import _size_2_t
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from anomalib.data import InferenceBatch
from anomalib.models.components import DynamicBufferMixin
from anomalib.models.components.feature_extractors import dryrun_find_featuremap_dims

from .anomaly_map import AnomalyMapGenerator

SUPPORTED_BACKBONES = ("vgg19_bn", "resnet18", "wide_resnet50_2", "efficientnet_b5")


def get_return_nodes(backbone: str) -> list[str]:
    """Get the return nodes for feature extraction from a backbone network.

    Args:
        backbone (str): Name of the backbone CNN. Must be one of
            ``{"resnet18", "wide_resnet50_2", "vgg19_bn", "efficientnet_b5"}``.

    Raises:
        NotImplementedError: If ``backbone`` is "efficientnet_b5".
        ValueError: If ``backbone`` is not one of the supported backbones.

    Returns:
        list[str]: List of layer names to extract features from.

    Example:
        >>> nodes = get_return_nodes("resnet18")
        >>> print(nodes)
        ['layer1', 'layer2', 'layer3']
    """
    if backbone == "efficientnet_b5":
        msg = "EfficientNet feature extractor has not implemented yet."
        raise NotImplementedError(msg)

    return_nodes: list[str]
    if backbone in {"resnet18", "wide_resnet50_2"}:
        return_nodes = ["layer1", "layer2", "layer3"]
    elif backbone == "vgg19_bn":
        return_nodes = ["features.25", "features.38", "features.52"]
    else:
        msg = f"Backbone {backbone} is not supported. Supported backbones are {SUPPORTED_BACKBONES}."
        raise ValueError(msg)
    return return_nodes


# TODO(samet-akcay): Replace this with the new torchfx feature extractor.
# CVS-122673
def get_feature_extractor(backbone: str, return_nodes: list[str]) -> GraphModule:
    """Create a feature extractor from a backbone CNN.

    Args:
        backbone (str): Name of the backbone CNN network.
        return_nodes (list[str]): List of layer names to extract features from.

    Raises:
        NotImplementedError: When ``backbone`` is efficientnet_b5.
        ValueError: When ``backbone`` is not supported.

    Returns:
        GraphModule: Feature extractor module.

    Example:
        >>> nodes = ["layer1", "layer2", "layer3"]
        >>> extractor = get_feature_extractor("resnet18", nodes)
        >>> x = torch.randn(1, 3, 224, 224)
        >>> features = extractor(x)
    """
    model = getattr(torchvision.models, backbone)(pretrained=True)
    feature_extractor = create_feature_extractor(model=model, return_nodes=return_nodes)
    feature_extractor.eval()

    return feature_extractor


class CfaModel(DynamicBufferMixin):
    """Torch implementation of the CFA Model.

    The model learns discriminative features by adapting them to coupled
    hyperspheres in the feature space. It uses a teacher-student architecture
    where the teacher network extracts features from normal samples to guide the
    student network.

    Args:
        backbone (str): Name of the backbone CNN network.
        gamma_c (int): Weight for centroid loss.
        gamma_d (int): Weight for distance loss.
        num_nearest_neighbors (int): Number of nearest neighbors for score
            computation.
        num_hard_negative_features (int): Number of hard negative features to use.
        radius (float): Initial radius of the hypersphere decision boundary.

    Example:
        >>> model = CfaModel(
        ...     backbone="resnet18",
        ...     gamma_c=1,
        ...     gamma_d=1,
        ...     num_nearest_neighbors=3,
        ...     num_hard_negative_features=3,
        ...     radius=0.5
        ... )
    """

    def __init__(
        self,
        backbone: str,
        gamma_c: int,
        gamma_d: int,
        num_nearest_neighbors: int,
        num_hard_negative_features: int,
        radius: float,
    ) -> None:
        super().__init__()
        self.gamma_c = gamma_c
        self.gamma_d = gamma_d

        self.num_nearest_neighbors = num_nearest_neighbors
        self.num_hard_negative_features = num_hard_negative_features

        self.register_buffer("memory_bank", torch.tensor(0.0))
        self.memory_bank: torch.Tensor

        self.backbone = backbone
        return_nodes = get_return_nodes(backbone)
        self.feature_extractor = get_feature_extractor(backbone, return_nodes)

        self.descriptor = Descriptor(self.gamma_d, backbone)
        self.radius = torch.ones(1, requires_grad=True) * radius

        self.anomaly_map_generator = AnomalyMapGenerator(
            num_nearest_neighbors=num_nearest_neighbors,
        )

    def get_scale(self, input_size: tuple[int, int] | torch.Size) -> torch.Size:
        """Get the scale of the feature maps.

        Args:
            input_size (tuple[int, int] | torch.Size): Input image dimensions
                (height, width).

        Returns:
            torch.Size: Feature map dimensions.

        Example:
            >>> model = CfaModel(...)
            >>> scale = model.get_scale((256, 256))
        """
        feature_map_metadata = dryrun_find_featuremap_dims(
            feature_extractor=self.feature_extractor,
            input_size=input_size,
            layers=get_return_nodes(self.backbone),
        )
        # Scale is to get the largest feature map dimensions of different layers
        # of the feature extractor. In a typical feature extractor, the first
        # layer has the highest resolution.
        resolution = next(iter(feature_map_metadata.values()))["resolution"]
        if isinstance(resolution, int):
            scale = (resolution,) * 2
        elif isinstance(resolution, tuple):
            scale = resolution
        else:
            msg = f"Unknown type {type(resolution)} for `resolution`. Expected types are either int or tuple[int, int]."
            raise TypeError(msg)
        return scale

    def initialize_centroid(self, data_loader: DataLoader) -> None:
        """Initialize the centroid of the memory bank.

        Computes the average feature representation of normal samples to
        initialize the memory bank centroids.

        Args:
            data_loader (DataLoader): DataLoader containing normal training
                samples.

        Example:
            >>> from torch.utils.data import DataLoader
            >>> model = CfaModel(...)
            >>> train_loader = DataLoader(...)
            >>> model.initialize_centroid(train_loader)
        """
        device = next(self.feature_extractor.parameters()).device
        with torch.no_grad():
            for i, data in enumerate(tqdm(data_loader)):
                batch = data.image.to(device)
                features = self.feature_extractor(batch)
                features = list(features.values())
                target_features = self.descriptor(features)
                self.memory_bank = ((self.memory_bank * i) + target_features.mean(dim=0, keepdim=True)) / (i + 1)

        self.memory_bank = rearrange(self.memory_bank, "b c h w -> (b h w) c")

        scale = self.get_scale(batch.shape[-2:])

        if self.gamma_c > 1:
            # TODO(samet-akcay): Create PyTorch KMeans class.
            # CVS-122673
            k_means = KMeans(n_clusters=(scale[0] * scale[1]) // self.gamma_c, max_iter=3000)
            cluster_centers = k_means.fit(self.memory_bank.cpu()).cluster_centers_
            self.memory_bank = torch.tensor(cluster_centers, requires_grad=False).to(device)

        self.memory_bank = rearrange(self.memory_bank, "h w -> w h")

    def compute_distance(self, target_oriented_features: torch.Tensor) -> torch.Tensor:
        """Compute distances between features and memory bank centroids.

        Args:
            target_oriented_features (torch.Tensor): Features from the descriptor
                network.

        Returns:
            torch.Tensor: Distance tensor.

        Example:
            >>> model = CfaModel(...)
            >>> features = torch.randn(32, 256, 32, 32)  # B x C x H x W
            >>> distances = model.compute_distance(features)
        """
        if target_oriented_features.ndim == 4:
            target_oriented_features = rearrange(target_oriented_features, "b c h w -> b (h w) c")

        features = target_oriented_features.pow(2).sum(dim=2, keepdim=True)
        centers = self.memory_bank.pow(2).sum(dim=0, keepdim=True).to(features.device)
        f_c = 2 * torch.matmul(target_oriented_features, (self.memory_bank.to(features.device)))
        return features + centers - f_c

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """Forward pass through the model.

        Args:
            input_tensor (torch.Tensor): Input image tensor.

        Raises:
            ValueError: When the memory bank is not initialized.

        Returns:
            torch.Tensor | InferenceBatch: During training, returns distance
                tensor. During inference, returns anomaly predictions.

        Example:
            >>> model = CfaModel(...)
            >>> x = torch.randn(32, 3, 256, 256)
            >>> predictions = model(x)
        """
        if self.memory_bank.ndim == 0:
            msg = "Memory bank is not initialized. Run `initialize_centroid` method first."
            raise ValueError(msg)

        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
            features = list(features.values())

        target_features = self.descriptor(features)
        distance = self.compute_distance(target_features)

        if self.training:
            return distance

        anomaly_map = self.anomaly_map_generator(
            distance=distance,
            scale=target_features.shape[-2:],
            image_size=input_tensor.shape[-2:],
        ).squeeze()
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)


class Descriptor(nn.Module):
    """Descriptor network that generates target-oriented features.

    Args:
        gamma_d (int): Weight for distance loss.
        backbone (str): Name of the backbone CNN network.

    Raises:
        ValueError: If ``backbone`` is not supported.

    Example:
        >>> descriptor = Descriptor(gamma_d=1, backbone="resnet18")
        >>> features = [torch.randn(32, 64, 64, 64)]
        >>> target_features = descriptor(features)
    """

    def __init__(self, gamma_d: int, backbone: str) -> None:
        super().__init__()

        self.backbone = backbone
        if self.backbone not in SUPPORTED_BACKBONES:
            msg = f"Supported backbones are {SUPPORTED_BACKBONES}. Got {self.backbone} instead."
            raise ValueError(msg)

        # TODO(samet-akcay): Automatically infer the number of dims
        # CVS-122673
        backbone_dims = {"vgg19_bn": 1280, "resnet18": 448, "wide_resnet50_2": 1792, "efficientnet_b5": 568}
        dim = backbone_dims[backbone]
        out_channels = 2 * dim // gamma_d if backbone == "efficientnet_b5" else dim // gamma_d

        self.layer = CoordConv2d(in_channels=dim, out_channels=out_channels, kernel_size=1)

    def forward(self, features: list[torch.Tensor] | dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the descriptor network.

        Args:
            features (list[torch.Tensor] | dict[str, torch.Tensor]): Features
                from the backbone network.

        Returns:
            torch.Tensor: Target-oriented features.

        Example:
            >>> descriptor = Descriptor(gamma_d=1, backbone="resnet18")
            >>> features = [torch.randn(32, 64, 64, 64)]
            >>> target_features = descriptor(features)
        """
        if isinstance(features, dict):
            features = list(features.values())

        patch_features: torch.Tensor | None = None
        for feature in features:
            pooled_features = (
                F.avg_pool2d(feature, 3, 1, 1) / feature.size(1)
                if self.backbone == "efficientnet_b5"
                else F.avg_pool2d(feature, 3, 1, 1)
            )
            patch_features = (
                pooled_features
                if patch_features is None
                else torch.cat((patch_features, F.interpolate(feature, patch_features.size(2), mode="bilinear")), dim=1)
            )

        return self.layer(patch_features)


class CoordConv2d(nn.Conv2d):
    """CoordConv layer that adds coordinate channels to input features.

    Implementation based on the paper "An Intriguing Failing of Convolutional
    Neural Networks and the CoordConv Solution".

    MIT License
    Copyright (c) 2018 Walsvid

    Paper: https://arxiv.org/abs/1807.03247
    Code: https://github.com/walsvid/CoordConv

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (_size_2_t): Size of the convolution kernel.
        stride (_size_2_t, optional): Stride of the convolution.
            Defaults to ``1``.
        padding (str | _size_2_t, optional): Padding added to input.
            Defaults to ``0``.
        dilation (_size_2_t, optional): Dilation of the kernel.
            Defaults to ``1``.
        groups (int, optional): Number of blocked connections. Defaults to ``1``.
        bias (bool, optional): If True, adds learnable bias. Defaults to ``True``.
        with_r (bool, optional): If True, adds radial coordinate channel.
            Defaults to ``False``.

    Example:
        >>> conv = CoordConv2d(64, 128, kernel_size=3)
        >>> x = torch.randn(32, 64, 32, 32)
        >>> out = conv(x)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        with_r: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        # AddCoord layer.
        self.add_coords = AddCoords(with_r)

        # Create conv layer on top of add_coords layer.
        self.conv2d = nn.Conv2d(
            in_channels=in_channels + 2 + int(with_r),  # 2 for rank-2, 1 for r
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CoordConv layer.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying coordinates and
                convolution.

        Example:
            >>> conv = CoordConv2d(64, 128, kernel_size=3)
            >>> x = torch.randn(32, 64, 32, 32)
            >>> out = conv(x)
        """
        out = self.add_coords(input_tensor)
        return self.conv2d(out)


class AddCoords(nn.Module):
    """Module that adds coordinate channels to input tensor.

    MIT License
    Copyright (c) 2018 Walsvid

    Paper: https://arxiv.org/abs/1807.03247
    Code: https://github.com/walsvid/CoordConv

    Args:
        with_r (bool, optional): If True, adds radial coordinate channel.
            Defaults to ``False``.

    Example:
        >>> coord_adder = AddCoords()
        >>> x = torch.randn(32, 64, 32, 32)
        >>> out = coord_adder(x)  # adds x,y coordinate channels
    """

    def __init__(self, with_r: bool = False) -> None:
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Add coordinate channels to input tensor.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with added coordinate channels.

        Example:
            >>> coord_adder = AddCoords()
            >>> x = torch.randn(32, 64, 32, 32)
            >>> out = coord_adder(x)  # adds x,y coordinate channels
        """
        # NOTE: This is a modified version of the original implementation,
        #   which only supports rank 2 tensors.
        batch, _, x_dim, y_dim = input_tensor.shape
        xx_ones = torch.ones([1, 1, 1, y_dim], dtype=torch.int32)
        yy_ones = torch.ones([1, 1, 1, x_dim], dtype=torch.int32)

        xx_range = torch.arange(x_dim, dtype=torch.int32)
        yy_range = torch.arange(y_dim, dtype=torch.int32)
        xx_range = xx_range[None, None, :, None]
        yy_range = yy_range[None, None, :, None]

        xx_channel = torch.matmul(xx_range, xx_ones)
        yy_channel = torch.matmul(yy_range, yy_ones)

        # Transpose y
        yy_channel = yy_channel.permute(0, 1, 3, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch, 1, 1, 1).to(input_tensor.device)
        yy_channel = yy_channel.repeat(batch, 1, 1, 1).to(input_tensor.device)

        out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr_channel = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            out = torch.cat([out, rr_channel], dim=1)

        return out
