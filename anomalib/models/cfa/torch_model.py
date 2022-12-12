"""Torch Implementatation of the CFA Model.

CFA: Coupled-hypersphere-based Feature Adaptation for Target-Oriented Anomaly Localization

Paper https://arxiv.org/abs/2206.04325
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from sklearn.cluster import KMeans
from torch import Tensor, nn
from torch.fx.graph_module import GraphModule
from torch.nn.common_types import _size_2_t
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from anomalib.models.components import GaussianBlur2d

SUPPORTED_BACKBONES = ("vgg19_bn", "resnet18", "wide_resnet50_2", "efficientnet_b5")


# TODO: Replace this with the new torchfx feature extractor.
def get_feature_extractor(backbone: str) -> GraphModule:
    """Get the feature extractor from the backbone CNN.

    Args:
        backbone (str): Backbone CNN network

    Raises:
        NotImplementedError: When the backbone is efficientnet_b5
        ValueError: When the backbone is not supported

    Returns:
        GraphModule: Feature extractor.
    """
    if backbone == "efficientnet_b5":
        raise NotImplementedError("EfficientNet feature extractor has not implemented yet.")

    return_nodes: List[str]
    if backbone in ("resnet18", "wide_resnet50_2"):
        return_nodes = ["layer1", "layer2", "layer3"]
    elif backbone == "vgg19_bn":
        return_nodes = ["features.25", "features.38", "features.52"]
    else:
        raise ValueError(f"Backbone {backbone} is not supported. Supported backbones are {SUPPORTED_BACKBONES}.")

    model = getattr(torchvision.models, backbone)(pretrained=True)
    feature_extractor = create_feature_extractor(model=model, return_nodes=return_nodes)
    feature_extractor.eval()

    return feature_extractor


class CfaModel(nn.Module):
    """Torch implementation of the CFA Model.

    Args:
        backbone (str): Backbone CNN network.
        gamma_c (int): gamma_c parameter from the paper.
        gamma_d (int): gamma_d parameter from the paper.
    """

    def __init__(
        self,
        backbone: str,
        gamma_c: int,
        gamma_d: int,
    ) -> None:
        super().__init__()
        self.gamma_c = gamma_c
        self.gamma_d = gamma_d

        self.input_size: Tuple[int, int]
        self.scale: int

        self.num_nearest_neighbors = 3
        self.num_hard_negative_features = 3

        self.feature_extractor = get_feature_extractor(backbone)
        self.memory_bank = torch.tensor(0, requires_grad=False)
        self.descriptor = Descriptor(self.gamma_d, backbone)
        self.radius = torch.ones(1, requires_grad=True) * 1e-5

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

                # TODO <<< Find a better place for these.
                self.input_size = (batch.size(2), batch.size(3))
                self.scale = features[0].size(2)
                # TODO >>> Find a better place for these.

                target_features = self.descriptor(features)
                self.memory_bank = ((self.memory_bank * i) + target_features.mean(dim=0, keepdim=True)) / (i + 1)

        self.memory_bank = rearrange(self.memory_bank, "b c h w -> (b h w) c")

        if self.gamma_c > 1:
            # TODO: Create PyTorch KMeans class.
            k_means = KMeans(n_clusters=(self.scale**2) // self.gamma_c, max_iter=3000)
            cluster_centers = k_means.fit(self.memory_bank.cpu()).cluster_centers_
            self.memory_bank = torch.tensor(cluster_centers, requires_grad=False).to(device)

        self.memory_bank = rearrange(self.memory_bank, "h w -> w h")

    def compute_distance(self, target_oriented_features: Tensor) -> Tensor:
        """Compute distance using target oriented features.

        Args:
            target_oriented_features (Tensor): Target oriented features computed
                using the descriptor.

        Returns:
            Tensor: Distance tensor.
        """
        if target_oriented_features.ndim == 4:
            target_oriented_features = rearrange(target_oriented_features, "b c h w -> b (h w) c")

        features = target_oriented_features.pow(2).sum(dim=2, keepdim=True)
        centers = self.memory_bank.pow(2).sum(dim=0, keepdim=True)
        f_c = 2 * torch.matmul(target_oriented_features, (self.memory_bank))
        distance = features + centers - f_c
        return distance

    def compute_loss(self, distance: Tensor) -> Tensor:
        """Compute the CFA loss.

        Args:
            distance (Tensor): Distance computed using target oriented features.

        Returns:
            Tensor: CFA loss.
        """
        num_neighbors = self.num_nearest_neighbors + self.num_hard_negative_features
        distance = distance.topk(num_neighbors, largest=False).values

        score = distance[:, :, : self.num_nearest_neighbors] - (self.radius**2).to(distance.device)
        l_att = torch.mean(torch.max(torch.zeros_like(score), score))

        score = (self.radius**2).to(distance.device) - distance[:, :, self.num_hard_negative_features :]
        l_rep = torch.mean(torch.max(torch.zeros_like(score), score - 0.1))

        loss = (l_att + l_rep) * 1000

        return loss

    def compute_score(self, distance: Tensor) -> Tensor:
        """Compute score based on the distance.

        Args:
            distance (Tensor): Distance tensor computed using target oriented
                features.

        Returns:
            Tensor: Score value.
        """
        distance = torch.sqrt(distance)

        n_neighbors = self.num_nearest_neighbors
        distance = distance.topk(n_neighbors, largest=False).values

        distance = (F.softmin(distance, dim=-1)[:, :, 0]) * distance[:, :, 0]
        distance = distance.unsqueeze(-1)

        score = rearrange(distance, "b (h w) c -> b c h w", h=self.scale)
        return score.detach()

    def compute_anomaly_map(self, score: Tensor) -> Tensor:
        """Compute anomaly map based on the score.

        Args:
            score (Tensor): Score tensor.

        Returns:
            Tensor: Anomaly map.
        """
        anomaly_map = score.mean(dim=1, keepdim=True)
        anomaly_map = F.interpolate(anomaly_map, size=self.input_size, mode="bilinear", align_corners=False)

        gaussian_blur = GaussianBlur2d(sigma=4).to(score.device)
        anomaly_map = gaussian_blur(anomaly_map)  # pylint: disable=not-callable
        return anomaly_map

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input_tensor (Tensor): Input tensor.

        Raises:
            ValueError: When the memory bank is not initialized.

        Returns:
            Tensor: Loss or anomaly map depending on the train/eval mode.
        """
        if self.memory_bank.ndim == 0:
            raise ValueError("Memory bank is not initialized. Run `initialize_centroid` method first.")

        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        target_features = self.descriptor(features)
        distance = self.compute_distance(target_features)

        if self.training:
            output = self.compute_loss(distance)
        else:
            score = self.compute_score(distance)
            output = self.compute_anomaly_map(score)

        return output


class Descriptor(nn.Module):
    """Descriptor module."""

    def __init__(self, gamma_d: int, backbone: str) -> None:
        super().__init__()

        self.backbone = backbone
        if self.backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"Supported backbones are {SUPPORTED_BACKBONES}. Got {self.backbone} instead.")

        # TODO: Automatically infer the number of dims
        backbone_dims = {"vgg19_bn": 1280, "resnet18": 448, "wide_resnet50_2": 1792, "efficientnet_b5": 568}
        dim = backbone_dims[backbone]
        out_channels = 2 * dim // gamma_d if backbone == "efficientnet_b5" else dim // gamma_d

        self.layer = CoordConv2d(in_channels=dim, out_channels=out_channels, kernel_size=1)

    def forward(self, features: Union[List[Tensor], Dict[str, Tensor]]) -> Tensor:
        """Forward pass."""
        if isinstance(features, dict):
            features = list(features.values())

        patch_features: Optional[Tensor] = None
        for i in features:
            i = F.avg_pool2d(i, 3, 1, 1) / i.size(1) if self.backbone == "efficientnet_b5" else F.avg_pool2d(i, 3, 1, 1)
            patch_features = (
                i
                if patch_features is None
                else torch.cat((patch_features, F.interpolate(i, patch_features.size(2), mode="bilinear")), dim=1)
            )

        target_oriented_features = self.layer(patch_features)
        return target_oriented_features


class CoordConv2d(nn.Conv2d):
    """CoordConv layer as in the paper.

    Link to the paper: https://arxiv.org/abs/1807.03247
    Link to the PyTorch implementation: https://github.com/walsvid/CoordConv
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
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

    def forward(self, input_tensor: Tensor) -> Tensor:  # pylint: disable=arguments-renamed
        """Forward pass.

        Args:
            input_tensor (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the CoordConv layer.
        """
        out = self.add_coords(input_tensor)
        out = self.conv2d(out)
        return out


class AddCoords(nn.Module):
    """Add coords to a tensor.

    Link to the paper: https://arxiv.org/abs/1807.03247
    Link to the PyTorch implementation: https://github.com/walsvid/CoordConv
    """

    def __init__(self, with_r: bool = False) -> None:
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input_tensor (Tensor): Input tensor

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
