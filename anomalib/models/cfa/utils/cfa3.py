import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from sklearn.cluster import KMeans
from torch import Tensor
from tqdm import tqdm
from utils.coordconv import CoordConv2d

from .metric import *


class DSVDD(nn.Module):
    def __init__(self, model, data_loader, cnn, gamma_c, gamma_d, device):
        super(DSVDD, self).__init__()
        self.device = device

        self.memory_bank = 0
        self.nu = 1e-3
        self.scale = None

        self.gamma_c = gamma_c
        self.gamma_d = gamma_d
        self.alpha = 1e-1
        self.num_nearest_neighbors = 3
        self.num_hard_negative_features = 3

        self.radius = nn.Parameter(1e-5 * torch.ones(1), requires_grad=True)
        self.descriptor = Descriptor(self.gamma_d, cnn).to(device)
        self._init_centroid(model, data_loader)
        self.memory_bank = rearrange(self.memory_bank, "b c h w -> (b h w) c").detach()

        if self.gamma_c > 1:
            self.memory_bank = self.memory_bank.cpu().detach().numpy()
            n_clusters = (self.scale**2) // self.gamma_c
            self.memory_bank = KMeans(n_clusters=n_clusters, max_iter=3000).fit(self.memory_bank).cluster_centers_
            self.memory_bank = torch.Tensor(self.memory_bank).to(device)

        self.memory_bank = self.memory_bank.transpose(-1, -2).detach()
        self.memory_bank = nn.Parameter(self.memory_bank, requires_grad=False)

    def _init_centroid(self, feature_extractor, data_loader):
        for i, (x, _, _) in enumerate(tqdm(data_loader)):
            x = x.to(self.device)
            patch_features = feature_extractor(x)
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

    def compute_loss(self, target_oriented_features: Tensor) -> Tensor:
        distance = self.compute_distance(target_oriented_features)

        n_neighbors = self.num_nearest_neighbors + self.num_hard_negative_features
        distance = distance.topk(n_neighbors, largest=False).values

        score = distance[:, :, : self.num_nearest_neighbors] - self.radius**2
        L_att = (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(score), score))

        score = self.radius**2 - distance[:, :, self.num_hard_negative_features :]
        L_rep = (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(score), score - self.alpha))

        loss = L_att + L_rep

        return loss

    def forward(self, patch_features: Tensor):
        target_oriented_features = self.descriptor(patch_features)
        target_oriented_features = rearrange(target_oriented_features, "b c h w -> b (h w) c")

        distance = self.compute_distance(target_oriented_features)
        distance = torch.sqrt(distance)

        n_neighbors = self.num_nearest_neighbors
        distance = distance.topk(n_neighbors, largest=False).values

        distance = (F.softmin(distance, dim=-1)[:, :, 0]) * distance[:, :, 0]
        distance = distance.unsqueeze(-1)

        score = rearrange(distance, "b (h w) c -> b c h w", h=self.scale)

        loss = 0
        if self.training:
            loss = self.compute_loss(target_oriented_features)

        return loss, score


class Descriptor(nn.Module):
    def __init__(self, gamma_d, cnn):
        super(Descriptor, self).__init__()
        self.cnn = cnn
        if cnn == "wrn50_2":
            dim = 1792
            self.layer = CoordConv2d(dim, dim // gamma_d, 1)
        elif cnn == "res18":
            dim = 448
            self.layer = CoordConv2d(dim, dim // gamma_d, 1)
        elif cnn == "effnet-b5":
            dim = 568
            self.layer = CoordConv2d(dim, 2 * dim // gamma_d, 1)
        elif cnn == "vgg19":
            dim = 1280
            self.layer = CoordConv2d(dim, dim // gamma_d, 1)

    def forward(self, p):
        sample = None
        for o in p:
            o = F.avg_pool2d(o, 3, 1, 1) / o.size(1) if self.cnn == "effnet-b5" else F.avg_pool2d(o, 3, 1, 1)
            sample = (
                o if sample is None else torch.cat((sample, F.interpolate(o, sample.size(2), mode="bilinear")), dim=1)
            )

        phi_p = self.layer(sample)
        return phi_p
