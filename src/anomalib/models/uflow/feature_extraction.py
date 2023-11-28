from typing import Tuple

import timm
import torch
import torch.nn.functional as F
from torch import nn

from anomalib.models.components.feature_extractors import TimmFeatureExtractor

AVAILABLE_EXTRACTORS = ["mcait", "resnet18", "wide_resnet50_2"]


def get_feature_extractor(backbone, input_size: Tuple[int, int] = (256, 256)):
    """
    Get feature extractor. Currently, is restricted to AVAILABLE_EXTRACTORS.
    Args:
        backbone (str): Backbone name.
        input_size (tuple[int, int]): Input size.

    Returns:
        FeatureExtractorInterface: Feature extractor.
    """
    assert backbone in AVAILABLE_EXTRACTORS, f"Feature extractor must be one of {AVAILABLE_EXTRACTORS}."
    if backbone in ["resnet18", "wide_resnet50_2"]:
        return FeatureExtractor(backbone, input_size, layers=["layer1", "layer2", "layer3"])
    elif backbone == "mcait":
        return MCaitFeatureExtractor()
    raise ValueError(
        "`backbone` must be one of `[mcait, resnet18, wide_resnet50_2]`. These are the only feature extractors tested. "
        "It does not mean that other feature extractors will not work."
    )


class FeatureExtractor(TimmFeatureExtractor):
    """Feature extractor based on ResNet (or others) backbones."""

    def __init__(self, backbone, input_size, layers=("layer1", "layer2", "layer3"), **kwargs):
        super(FeatureExtractor, self).__init__(backbone, layers, pre_trained=True, requires_grad=False)
        self.channels = self.feature_extractor.feature_info.channels()
        self.scale_factors = self.feature_extractor.feature_info.reduction()
        self.scales = range(len(self.scale_factors))

        self.feature_normalizations = nn.ModuleList()
        for in_channels, scale in zip(self.channels, self.scale_factors):
            self.feature_normalizations.append(
                nn.LayerNorm(
                    [in_channels, int(input_size[0] / scale), int(input_size[1] / scale)], elementwise_affine=True
                )
            )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, img, **kwargs):
        features = self.extract_features(img)
        normalized_features = self.normalize_features(features, **kwargs)
        return normalized_features

    def extract_features(self, img, **kwargs):
        self.feature_extractor.eval()
        return self.feature_extractor(img)

    def normalize_features(self, features, **kwargs):
        return [self.feature_normalizations[i](feature) for i, feature in enumerate(features)]


class MCaitFeatureExtractor(nn.Module):
    """
    Feature extractor based on MCait backbone. This is the proposed feature extractor in the paper. It uses two
    independently trained Cait models, at different scales, with input sizes 448 and 224, respectively.
    It also includes a normalization layer for each scale.
    """

    def __init__(self):
        super(MCaitFeatureExtractor, self).__init__()
        self.input_size = 448
        self.extractor1 = timm.create_model("cait_m48_448", pretrained=True)
        self.extractor2 = timm.create_model("cait_s24_224", pretrained=True)
        self.channels = [768, 384]
        self.scale_factors = [16, 32]
        self.scales = range(len(self.scale_factors))

        for param in self.extractor1.parameters():
            param.requires_grad = False
        for param in self.extractor2.parameters():
            param.requires_grad = False

    def forward(self, img, training=True):
        features = self.extract_features(img)
        normalized_features = self.normalize_features(features, training=training)
        return normalized_features

    def extract_features(self, img, **kwargs):
        self.extractor1.eval()
        self.extractor2.eval()

        # Scale 1 --> Extractor 1
        x1 = self.extractor1.patch_embed(img)
        x1 = x1 + self.extractor1.pos_embed
        x1 = self.extractor1.pos_drop(x1)
        for i in range(41):  # paper Table 6. Block Index = 40
            x1 = self.extractor1.blocks[i](x1)

        # Scale 2 --> Extractor 2
        img_sub = F.interpolate(torch.Tensor(img), size=(224, 224), mode="bicubic", align_corners=True)
        x2 = self.extractor2.patch_embed(img_sub)
        x2 = x2 + self.extractor2.pos_embed
        x2 = self.extractor2.pos_drop(x2)
        for i in range(21):
            x2 = self.extractor2.blocks[i](x2)

        features = [x1, x2]
        return features

    def normalize_features(self, features, **kwargs):
        normalized_features = []
        for i, extractor in enumerate([self.extractor1, self.extractor2]):
            batch, _, channels = features[i].shape
            scale_factor = self.scale_factors[i]

            x = extractor.norm(features[i].contiguous())
            x = x.permute(0, 2, 1)
            x = x.reshape(batch, channels, self.input_size // scale_factor, self.input_size // scale_factor)
            normalized_features.append(x)

        return normalized_features
