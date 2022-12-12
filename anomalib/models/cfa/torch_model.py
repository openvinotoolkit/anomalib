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
from utils.coordconv import CoordConv2d

from anomalib.models.cfa.utils.metric import *
from anomalib.models.components import GaussianBlur2d

SUPPORTED_BACKBONES = ("vgg19_bn", "resnet18", "wide_resnet50_2", "efficientnet_b5")


# TODO: >>> This is temporary.
def initialize_weights(m) -> None:
    torch.manual_seed(0)
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


# TODO: <<< This is temporary.


class CfaModel(nn.Module):
    def __init__(self, model, data_loader, cnn, gamma_c, gamma_d, device):
        super().__init__()
        self.device = device

        self.C = 0
        self.nu = 1e-3
        self.scale = None

        self.gamma_c = gamma_c
        self.gamma_d = gamma_d
        self.alpha = 1e-1
        self.K = 3
        self.J = 3

        self.r = nn.Parameter(1e-5 * torch.ones(1), requires_grad=True)
        self.Descriptor = Descriptor(self.gamma_d, cnn).to(device)
        # TODO: >>> Temporary.
        self.Descriptor.apply(initialize_weights)
        # TODO <<< Temporary.
        self._init_centroid(model, data_loader)
        self.C = rearrange(self.C, "b c h w -> (b h w) c").detach()

        if self.gamma_c > 1:
            self.C = self.C.cpu().detach().numpy()
            self.C = KMeans(n_clusters=(self.scale**2) // self.gamma_c, max_iter=3000).fit(self.C).cluster_centers_
            self.C = torch.Tensor(self.C).to(device)

        self.C = self.C.transpose(-1, -2).detach()
        self.C = nn.Parameter(self.C, requires_grad=False)

    def forward(self, p):
        phi_p = self.Descriptor(p)
        phi_p = rearrange(phi_p, "b c h w -> b (h w) c")

        features = torch.sum(torch.pow(phi_p, 2), 2, keepdim=True)
        centers = torch.sum(torch.pow(self.C, 2), 0, keepdim=True)
        f_c = 2 * torch.matmul(phi_p, (self.C))
        dist = features + centers - f_c
        dist = torch.sqrt(dist)

        n_neighbors = self.K
        dist = dist.topk(n_neighbors, largest=False).values

        dist = (F.softmin(dist, dim=-1)[:, :, 0]) * dist[:, :, 0]
        dist = dist.unsqueeze(-1)

        score = rearrange(dist, "b (h w) c -> b c h w", h=self.scale)

        loss = 0
        if self.training:
            loss = self._soft_boundary(phi_p)

        return loss, score

    def _soft_boundary(self, phi_p):
        features = torch.sum(torch.pow(phi_p, 2), 2, keepdim=True)
        centers = torch.sum(torch.pow(self.C, 2), 0, keepdim=True)
        f_c = 2 * torch.matmul(phi_p, (self.C))
        dist = features + centers - f_c
        n_neighbors = self.K + self.J
        dist = dist.topk(n_neighbors, largest=False).values

        score = dist[:, :, : self.K] - self.r**2
        L_att = (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(score), score))

        score = self.r**2 - dist[:, :, self.J :]
        L_rep = (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(score), score - self.alpha))

        loss = L_att + L_rep

        return loss

    def _init_centroid(self, model, data_loader):
        for i, (x, _, _) in enumerate(tqdm(data_loader)):
            x = x.to(self.device)
            p = model(x)
            self.scale = p[0].size(2)
            phi_p = self.Descriptor(p)
            self.C = ((self.C * i) + torch.mean(phi_p, dim=0, keepdim=True).detach()) / (i + 1)


class Descriptor(nn.Module):
    def __init__(self, gamma_d, cnn):
        super(Descriptor, self).__init__()
        self.cnn = cnn
        if cnn == "wide_resnet50_2":
            dim = 1792
            self.layer = CoordConv2d(dim, dim // gamma_d, 1)
        elif cnn == "resnet18":
            dim = 448
            self.layer = CoordConv2d(dim, dim // gamma_d, 1)
        elif cnn == "efficientnet_b5":
            dim = 568
            self.layer = CoordConv2d(dim, 2 * dim // gamma_d, 1)
        elif cnn == "vgg19_bn":
            dim = 1280
            self.layer = CoordConv2d(dim, dim // gamma_d, 1)

    def forward(self, p):
        sample = None
        for o in p:
            o = F.avg_pool2d(o, 3, 1, 1) / o.size(1) if self.cnn == "efficientnet_b5" else F.avg_pool2d(o, 3, 1, 1)
            sample = (
                o if sample is None else torch.cat((sample, F.interpolate(o, sample.size(2), mode="bilinear")), dim=1)
            )

        phi_p = self.layer(sample)
        return phi_p


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
