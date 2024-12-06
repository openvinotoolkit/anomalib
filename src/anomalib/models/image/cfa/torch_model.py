"""Torch Implementatation of the CFA Model.

CFA: Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization

Paper https://arxiv.org/abs/2206.04325
"""

# Copyright (C) 2022-2024 Intel Corporation
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

from anomalib.models.components import DynamicBufferMixin
from anomalib.models.components.feature_extractors import dryrun_find_featuremap_dims

from .anomaly_map import AnomalyMapGenerator

SUPPORTED_BACKBONES = ("vgg19_bn", "resnet18", "wide_resnet50_2", "efficientnet_b5")


def get_return_nodes(backbone: str) -> list[str]:
    """Get the return nodes for a given backbone.

    Args:
        backbone (str): The name of the backbone. Must be one of
            {"resnet18", "wide_resnet50_2", "vgg19_bn", "efficientnet_b5"}.

    Raises:
        NotImplementedError: If the backbone is "efficientnet_b5".
        ValueError: If the backbone is not one of the supported backbones.

    Returns:
        list[str]: A list of return nodes for the given backbone.
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
    """Get the feature extractor from the backbone CNN.

    Args:
        backbone (str): Backbone CNN network
        return_nodes (list[str]): A list of return nodes for the given backbone.

    Raises:
        NotImplementedError: When the backbone is efficientnet_b5
        ValueError: When the backbone is not supported

    Returns:
        GraphModule: Feature extractor.
    """
    model = getattr(torchvision.models, backbone)(pretrained=True)
    feature_extractor = create_feature_extractor(model=model, return_nodes=return_nodes)
    feature_extractor.eval()

    return feature_extractor


class CfaModel(DynamicBufferMixin):
    """Torch implementation of the CFA Model.

    Args:
        backbone (str): Backbone CNN network.
        gamma_c (int): gamma_c parameter from the paper.
        gamma_d (int): gamma_d parameter from the paper.
        num_nearest_neighbors (int): Number of nearest neighbors.
        num_hard_negative_features (int): Number of hard negative features.
        radius (float): Radius of the hypersphere to search the soft boundary.
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
        """Get the scale of the feature map.

        Args:
            input_size (tuple[int, int]): Input size of the image tensor.
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
        """Initialize the Centroid of the Memory Bank.

        Args:
            data_loader (DataLoader):  Train Dataloader.

        Returns:
            Tensor: Memory Bank.
        """
        device = next(self.feature_extractor.parameters()).device
        with torch.no_grad():
            for i, data in enumerate(tqdm(data_loader)):
                batch = data["image"].to(device)
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
        """Compute distance using target oriented features.

        Args:
            target_oriented_features (torch.Tensor): Target oriented features computed
                using the descriptor.

        Returns:
            Tensor: Distance tensor.
        """
        if target_oriented_features.ndim == 4:
            target_oriented_features = rearrange(target_oriented_features, "b c h w -> b (h w) c")

        features = target_oriented_features.pow(2).sum(dim=2, keepdim=True)
        centers = self.memory_bank.pow(2).sum(dim=0, keepdim=True).to(features.device)
        f_c = 2 * torch.matmul(target_oriented_features, (self.memory_bank.to(features.device)))
        return features + centers - f_c

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Raises:
            ValueError: When the memory bank is not initialized.

        Returns:
            Tensor: Loss or anomaly map depending on the train/eval mode.
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

        return (
            distance
            if self.training
            else self.anomaly_map_generator(
                distance=distance,
                scale=target_features.shape[-2:],
                image_size=input_tensor.shape[-2:],
            )
        )


class Descriptor(nn.Module):
    """Descriptor module."""

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
        """Forward pass."""
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
    """CoordConv layer as in the paper.

    MIT License
    Copyright (c) 2018 Walsvid

    Link to the paper: https://arxiv.org/abs/1807.03247
    Link to the PyTorch implementation: https://github.com/walsvid/CoordConv
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
            in_channels=in_channels + 2 + int(with_r),  # 2 for rank-2 tensor, 1 for r if with_r
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-renamed
        """Forward pass.

        Args:
            input_tensor (torch.Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the CoordConv layer.
        """
        out = self.add_coords(input_tensor)
        return self.conv2d(out)


class AddCoords(nn.Module):
    """Add coords to a tensor.

    MIT License
    Copyright (c) 2018 Walsvid

    Link to the paper: https://arxiv.org/abs/1807.03247
    Link to the PyTorch implementation: https://github.com/walsvid/CoordConv
    """

    def __init__(self, with_r: bool = False) -> None:
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_tensor (torch.Tensor): Input tensor

        Returns:
            Tensor: Output tensor with added coordinates.
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
