"""PyTorch model for the SuperSimpleNet model implementation."""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import Wide_ResNet50_2_Weights

from anomalib.models.components import TorchFXFeatureExtractor

# Original Code
# Copyright (c) 2024 Blaž Rolih
# https://github.com/blaz-r/SuperSimpleNet.
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


class SuperSimpleNet(nn.Module):
    """SuperSimpleNet Pytorch model.

    It consists of feature extractor, feature adaptor, anomaly generation mechanism and segmentation-detection module.

    """

    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor(backbone="wide_resnet50_2", layers=["layer2", "layer3"])

        channels = self.feature_extractor.get_channels_dim()
        self.adaptor = FeatureAdaptor(channels)
        self.segdec = SegmentationDetectionModule(input_dim=channels, stop_grad=True)

    def forward(self, input_tensor: torch.Tensor):
        features = self.feature_extractor(input_tensor)
        adapted = self.adaptor(features)

        if self.training:
            pass
        else:
            anomaly_map, anomaly_score = self.segdec(adapted)

            return anomaly_map, anomaly_score


def init_weights(m: nn.Module):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.constant_(m.weight, 1)


class FeatureExtractor(nn.Module):
    def __init__(self, backbone: str, layers: list[str], patch_size: int = 3):
        super().__init__()

        self.feature_extractor = TorchFXFeatureExtractor(
            backbone=backbone,
            return_nodes=layers,
            weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1,
        )
        self.pooler = nn.AvgPool2d(
            kernel_size=patch_size,
            stride=1,
            padding=patch_size // 2,
        )

    def forward(self, input_tensor: torch.Tensor):
        # extract features
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)

        features = list(features.values())

        _, _, h, w = features[0].shape
        feature_map = []
        for layer in features:
            # upscale all to 2x the size of the first (largest)
            resized = F.interpolate(
                layer,
                size=(h * 2, w * 2),
                mode="bilinear",
            )
            feature_map.append(resized)
        # channel-wise concat
        feature_map = torch.cat(feature_map, dim=1)

        # neighboring patch aggregation
        feature_map = self.pooler(feature_map)

        return feature_map

    def get_channels_dim(self) -> int:
        # dryrun
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(torch.rand(1, 3, 256, 256))
        # sum channels
        channels = sum(feature.shape[1] for feature in features.values())
        return channels


class FeatureAdaptor(nn.Module):
    def __init__(self, projection_dim: int):
        super().__init__()
        # linear layer equivalent
        self.projection = nn.Conv2d(
            in_channels=projection_dim,
            out_channels=projection_dim,
            kernel_size=1,
            stride=1,
        )
        self.apply(init_weights)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


class SegmentationDetectionModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        stop_grad: bool = False,
    ):
        super().__init__()
        self.stop_grad = stop_grad

        # 1x1 conv - linear layer equivalent
        self.seg = nn.Sequential(
            nn.Conv2d(
                in_channels=input_dim,
                out_channels=1024,
                kernel_size=1,
                stride=1,
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                in_channels=1024,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

        self.dec_block = nn.Sequential(
            nn.Conv2d(
                in_channels=input_dim + 1,
                out_channels=128,
                kernel_size=5,
                padding="same",
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.map_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.map_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        self.dec_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dec_max_pool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

        # 128 from dec and 2 from map, * 2 due to max and avg pool
        self.fc_score = nn.Linear(in_features=128 * 2 + 2, out_features=1)

        self.apply(init_weights)

    def get_params(self):
        seg_params = self.seg.parameters()
        dec_params = list(self.dec_block.parameters()) + list(self.fc_score.parameters())
        return seg_params, dec_params

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # get anomaly map from seg head
        map = self.seg(input)

        map_dec_copy = map
        if self.stop_grad:
            map_dec_copy = map_dec_copy.detach()
        # dec conv layer takes feat + map
        mask_cat = torch.cat((input, map_dec_copy), dim=1)
        dec_out = self.dec_block(mask_cat)

        # conv block result pooling
        dec_max = self.dec_max_pool(dec_out)
        dec_avg = self.dec_avg_pool(dec_out)

        # predicted map pooling (and stop grad)
        map_max = self.map_max_pool(map)
        if self.stop_grad:
            map_max = map_max.detach()

        map_avg = self.map_avg_pool(map)
        if self.stop_grad:
            map_avg = map_avg.detach()

        # final dec layer: conv channel max and avg and map max and avg
        dec_cat = torch.cat((dec_max, dec_avg, map_max, map_avg), dim=1).squeeze(
            dim=(2, 3),
        )
        score = self.fc_score(dec_cat).squeeze(dim=1)

        return map, score


if __name__ == "__main__":
    ssn = SuperSimpleNet()
    ssn.eval()
    x = torch.rand(1, 3, 256, 256)
    p_map, p_score = ssn(x)
    print(p_map.shape, p_score.shape)
