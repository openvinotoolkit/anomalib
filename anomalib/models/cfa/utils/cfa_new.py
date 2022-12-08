from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from sklearn.cluster import KMeans
from torch import Tensor
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
from utils.coordconv import CoordConv2d

from anomalib.models.cfa.anomaly_map import AnomalyMapGenerator

from .metric import *

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


class CfaModel(nn.Module):
    def __init__(self, data_loader, backbone, gamma_c, gamma_d, device):
        super().__init__()
        self.device = device
        self.feature_extractor = get_feature_extractor(backbone, device=device)

        self.memory_bank = 0
        self.nu = 1e-3
        self.scale = None

        self.gamma_c = gamma_c
        self.gamma_d = gamma_d
        self.alpha = 1e-1
        self.num_nearest_neighbors = 3
        self.num_hard_negative_features = 3

        self.radius = nn.Parameter(1e-5 * torch.ones(1), requires_grad=True)
        self.descriptor = Descriptor(self.gamma_d, backbone).to(device)
        self._init_centroid(self.feature_extractor, data_loader)
        self.memory_bank = rearrange(self.memory_bank, "b c h w -> (b h w) c").detach()

        if self.gamma_c > 1:
            self.memory_bank = self.memory_bank.cpu().detach().numpy()
            n_clusters = (self.scale**2) // self.gamma_c
            self.memory_bank = KMeans(n_clusters=n_clusters, max_iter=3000).fit(self.memory_bank).cluster_centers_
            self.memory_bank = torch.Tensor(self.memory_bank).to(device)

        self.memory_bank = self.memory_bank.transpose(-1, -2).detach()
        self.memory_bank = nn.Parameter(self.memory_bank, requires_grad=False)

        self.anomaly_map_generator = AnomalyMapGenerator(
            image_size=(224, 224), layer_size=self.scale, num_nearest_neighbors=self.num_nearest_neighbors, sigma=4
        )

    def _init_centroid(self, feature_extractor, data_loader) -> None:
        # for i, batch in enumerate(tqdm(data_loader)):
        for i, (x, _, _) in enumerate(tqdm(data_loader)):
            # x = batch["image"]
            x = x.to(self.device)
            patch_features = feature_extractor(x)
            patch_features = [value for value in patch_features.values()]
            self.scale = patch_features[0].size(2)
            target_oriented_features = self.descriptor(patch_features)
            self.memory_bank = (
                (self.memory_bank * i) + torch.mean(target_oriented_features, dim=0, keepdim=True).detach()
            ) / (i + 1)

    def compute_distance(self, target_oriented_features: Tensor) -> Tensor:
        features = torch.sum(torch.pow(target_oriented_features, 2), 2, keepdim=True)
        centers = torch.sum(torch.pow(self.memory_bank, 2), 0, keepdim=True)
        f_c = 2 * torch.matmul(target_oriented_features, (self.memory_bank))
        distance = features + centers - f_c
        return distance

    def compute_loss(self, distance: Tensor) -> Tensor:
        n_neighbors = self.num_nearest_neighbors + self.num_hard_negative_features
        distance = distance.topk(n_neighbors, largest=False).values

        score = distance[:, :, : self.num_nearest_neighbors] - self.radius**2
        L_att = (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(score), score))

        score = self.radius**2 - distance[:, :, self.num_hard_negative_features :]
        L_rep = (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(score), score - self.alpha))

        loss = L_att + L_rep

        return loss

    def forward(self, input_tensor: Tensor):
        self.feature_extractor.eval()
        with torch.no_grad():
            patch_features = self.feature_extractor(input_tensor)

        patch_features = [v for v in patch_features.values()]
        target_oriented_features = self.descriptor(patch_features)
        target_oriented_features = rearrange(target_oriented_features, "b c h w -> b (h w) c")

        distance = self.compute_distance(target_oriented_features)

        if self.training:
            output = self.compute_loss(distance)
        else:
            output = self.anomaly_map_generator(distance=distance)

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

    def forward(self, raw_patch_features: List[Tensor]) -> Tensor:
        patch_features: Optional[Tensor] = None
        for i in raw_patch_features:
            i = F.avg_pool2d(i, 3, 1, 1) / i.size(1) if self.backbone == "efficientnet_b5" else F.avg_pool2d(i, 3, 1, 1)
            patch_features = (
                i
                if patch_features is None
                else torch.cat((patch_features, F.interpolate(i, patch_features.size(2), mode="bilinear")), dim=1)
            )

        target_oriented_features = self.layer(patch_features)
        return target_oriented_features
