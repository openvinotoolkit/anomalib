"""Feature Extractor for U-Flow model."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

import timm
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from anomalib.models.components.feature_extractors import TimmFeatureExtractor

AVAILABLE_EXTRACTORS = ["mcait", "resnet18", "wide_resnet50_2"]


def get_feature_extractor(backbone: str, input_size: tuple[int, int] = (256, 256)) -> nn.Module:
    """Get feature extractor. Currently, is restricted to AVAILABLE_EXTRACTORS.

    Args:
        backbone (str): Backbone name.
        input_size (tuple[int, int]): Input size.

    Raises:
        ValueError if unknown backbone is provided.

    Returns:
        FeatureExtractorInterface: Feature extractor.
    """
    if backbone not in AVAILABLE_EXTRACTORS:
        msg = f"Feature extractor must be one of {AVAILABLE_EXTRACTORS}."
        raise ValueError(msg)

    feature_extractor: nn.Module
    if backbone in {"resnet18", "wide_resnet50_2"}:
        feature_extractor = FeatureExtractor(backbone, input_size, layers=("layer1", "layer2", "layer3")).eval()
    if backbone == "mcait":
        feature_extractor = MCaitFeatureExtractor().eval()

    return feature_extractor


class FeatureExtractor(TimmFeatureExtractor):
    """Feature extractor based on ResNet (or others) backbones.

    Args:
        backbone (str): Backbone of the feature extractor.
        input_size (tuple[int, int]): Input image size used for computing normalization layers.
        layers (tuple[str], optional): Layers from which to extract features.
            Defaults to ("layer1", "layer2", "layer3").
    """

    def __init__(
        self,
        backbone: str,
        input_size: tuple[int, int],
        layers: tuple[str, ...] = ("layer1", "layer2", "layer3"),
        **kwargs,  # noqa: ARG002 | unused argument
    ) -> None:
        super().__init__(backbone, layers, pre_trained=True, requires_grad=False)
        self.channels = self.feature_extractor.feature_info.channels()
        self.scale_factors = self.feature_extractor.feature_info.reduction()
        self.scales = range(len(self.scale_factors))

        self.feature_normalizations = nn.ModuleList()
        for in_channels, scale in zip(self.channels, self.scale_factors, strict=True):
            self.feature_normalizations.append(
                nn.LayerNorm(
                    [in_channels, int(input_size[0] / scale), int(input_size[1] / scale)],
                    elementwise_affine=True,
                ),
            )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Normalized features."""
        features = self.extract_features(img)
        return self.normalize_features(features)

    def extract_features(self, img: torch.Tensor) -> torch.Tensor:
        """Extract features."""
        self.feature_extractor.eval()
        return self.feature_extractor(img)

    def normalize_features(self, features: Iterable[torch.Tensor]) -> list[torch.Tensor]:
        """Normalize features."""
        return [self.feature_normalizations[i](feature) for i, feature in enumerate(features)]


class MCaitFeatureExtractor(nn.Module):
    """Feature extractor based on MCait backbone.

    This is the proposed feature extractor in the paper. It uses two
    independently trained Cait models, at different scales, with input sizes 448 and 224, respectively.
    It also includes a normalization layer for each scale.
    """

    def __init__(self) -> None:
        super().__init__()
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

    def forward(self, img: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Return normalized features."""
        features = self.extract_features(img)
        return self.normalize_features(features, training=training)

    def extract_features(self, img: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ARG002 | unused argument
        """Extract features from ``img`` from the two extractors.

        Args:
            img (torch.Tensor): Input image
            kwargs: unused

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Features from the two extractors.
        """
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

        return (x1, x2)

    def normalize_features(self, features: torch.Tensor, **kwargs) -> torch.Tensor:  # noqa: ARG002 | unused argument
        """Normalize features.

        Args:
            features (torch.Tensor): Features to normalize.
            **kwargs: unused

        Returns:
            torch.Tensor: Normalized features.
        """
        normalized_features = []
        for i, extractor in enumerate([self.extractor1, self.extractor2]):
            batch, _, channels = features[i].shape
            scale_factor = self.scale_factors[i]

            x = extractor.norm(features[i].contiguous())
            x = x.permute(0, 2, 1)
            x = x.reshape(batch, channels, self.input_size // scale_factor, self.input_size // scale_factor)
            normalized_features.append(x)

        return normalized_features
