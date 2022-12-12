"""Torch Model Implementation of the CFA Model."""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from sklearn.cluster import KMeans
from torch import Tensor
from torch.nn.common_types import _size_2_t
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from anomalib.models.components import GaussianBlur2d

SUPPORTED_BACKBONES = ("vgg19_bn", "resnet18", "wide_resnet50_2", "efficientnet_b5")


def get_feature_extractor(backbone: str, device: Optional[torch.device] = None):
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

    if device is not None:
        feature_extractor = feature_extractor.to(device)

    return feature_extractor


# TODO: >>> This is temporary.
def initialize_weights(m) -> None:
    torch.manual_seed(0)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


# TODO: <<< This is temporary.


class CfaModel(nn.Module):
    def __init__(
        self, data_loader: DataLoader, backbone: str, gamma_c: int, gamma_d: int, device: torch.device
    ) -> None:
        super().__init__()
        self.gamma_c = gamma_c
        self.gamma_d = gamma_d
        self.device = device

        self.input_size: Tuple[int, int]
        self.scale: int

        self.num_nearest_neighbors = 3
        self.num_hard_negative_features = 3

        self.feature_extractor = get_feature_extractor(backbone=backbone, device=device)
        self.descriptor = Descriptor(self.gamma_d, backbone).to(device)
        self.r = torch.ones(1, requires_grad=True) * 1e-5

        # TODO: >>> Temporary.
        self.descriptor.apply(initialize_weights)
        # TODO <<< Temporary.

        self.memory_bank = self._init_centroid(data_loader)

    def _init_centroid(self, data_loader: DataLoader) -> Tensor:
        """Initialize the Centroid of the Memory Bank.

        Args:
            model (nn.Module): Feature Extractor Model.
            data_loader (DataLoader): Train Dataloader.

        Returns:
            Tensor: Memory Bank.
        """
        memory_bank: Tensor = torch.tensor(0, requires_grad=False)
        with torch.no_grad():
            for i, data in enumerate(tqdm(data_loader)):
            # for i, (batch, _, _) in enumerate(tqdm(data_loader)):
                batch = data["image"].to(self.device)
                # batch = batch.to(self.device)
                features = self.feature_extractor(batch)
                # TODO: >>> Conversion from dict to list.
                features = [val for val in features.values()]
                # TODO <<< Conversion from dict to list.

                # TODO <<< Find a better place for these.
                self.input_size = (batch.size(2), batch.size(3))
                self.scale = features[0].size(2)
                # TODO >>> Find a better place for these.

                oriented_features = self.descriptor(features)
                memory_bank = ((memory_bank * i) + oriented_features.mean(dim=0, keepdim=True)) / (i + 1)

        memory_bank = rearrange(memory_bank, "b c h w -> (b h w) c")

        if self.gamma_c > 1:
            k_means = KMeans(n_clusters=(self.scale**2) // self.gamma_c, max_iter=3000)
            cluster_centers = k_means.fit(memory_bank.cpu()).cluster_centers_
            memory_bank = torch.tensor(cluster_centers, requires_grad=False).to(self.device)

        memory_bank = rearrange(memory_bank, "h w -> w h")

        return memory_bank

    def compute_distance(self, target_oriented_features: Tensor) -> Tensor:
        """Compute distance using target oriented features.

        Args:
            target_oriented_features (Tensor): Target oriented features computed
                using the descriptor.

        Returns:
            Tensor: Distance tensor.
        """
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

        score = distance[:, :, : self.num_nearest_neighbors] - (self.r**2).to(self.device)
        L_att = torch.mean(torch.max(torch.zeros_like(score), score))

        score = (self.r**2).to(self.device) - distance[:, :, self.num_hard_negative_features :]
        L_rep = torch.mean(torch.max(torch.zeros_like(score), score - 0.1))

        loss = (L_att + L_rep) * 1000

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
        anomaly_map = gaussian_blur(anomaly_map)
        return anomaly_map

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass of the CFA model.

        Args:
            input (Tensor): Input batch to the model.

        Returns:
            Tensor: Loss or the score values depending on train/eval modes.
        """
        self.feature_extractor.eval()
        features = self.feature_extractor(input)

        target_features = self.descriptor(features)
        target_features = rearrange(target_features, "b c h w -> b (h w) c")

        distance = self.compute_distance(target_features)

        if self.training:
            output = self.compute_loss(distance)
        else:
            score = self.compute_score(distance)
            output = self.compute_anomaly_map(score)
        return output


class Descriptor(nn.Module):
    def __init__(self, gamma_d: int, backbone: str) -> None:
        super(Descriptor, self).__init__()

        self.backbone = backbone
        if self.backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"Supported backbones are {SUPPORTED_BACKBONES}. Got {self.backbone} instead.")

        backbone_dims = {"vgg19_bn": 1280, "resnet18": 448, "wide_resnet50_2": 1792, "efficientnet_b5": 568}
        dim = backbone_dims[backbone]
        out_channels = 2 * dim // gamma_d if backbone == "efficientnet_b5" else dim // gamma_d

        self.layer = CoordConv2d(in_channels=dim, out_channels=out_channels, kernel_size=1)

    def forward(self, features: Union[List[Tensor], Dict[str, Tensor]]) -> Tensor:
        """Forward pass."""
        if isinstance(features, dict):
            features = [values for values in features.values()]

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

    def forward(self, input: Tensor) -> Tensor:
        out = self.add_coords(input)
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

    def forward(self, input: Tensor) -> Tensor:
        # NOTE: This is a modified version of the original implementation, which only supports rank 2 tensors.
        batch, _, x_dim, y_dim = input.shape
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

        xx_channel = xx_channel.repeat(batch, 1, 1, 1).to(input.device)
        yy_channel = yy_channel.repeat(batch, 1, 1, 1).to(input.device)

        out = torch.cat([input, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
            out = torch.cat([out, rr], dim=1)

        return out
