"""PyTorch model for the SuperSimpleNet model implementation."""
import torch
from torch import nn
import torch.nn.functional as F

from anomalib.models.components import TorchFXFeatureExtractor
from torchvision.models import Wide_ResNet50_2_Weights


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

    def __init__(self, patch_size: int):
        super().__init__()

        self.feature_extractor = TorchFXFeatureExtractor(
                backbone="wide_resnet50_2",
                return_nodes=["layer2", "layer3"],
                weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1
            )
        self.pooler = nn.AvgPool2d(
            kernel_size=patch_size, stride=1, padding=patch_size // 2
        )

    def get_features(self, input_tensor: torch.Tensor):
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
                layer, size=(h * 2, w * 2), mode="bilinear"
            )
            feature_map.append(resized)
        # channel-wise concat
        feature_map = torch.cat(feature_map, dim=1)

        # neighboring patch aggregation
        feature_map = self.pooler(feature_map)

        return feature_map

    def forward(self, input_tensor: torch.Tensor):
        features = self.get_features(input_tensor)

        return features


if __name__ == '__main__':
    ssn = SuperSimpleNet(patch_size=3)
    x = torch.rand(1, 3, 256, 256)
    y = ssn(x)
    print(y.shape)
