"""Feature extractor using timm models and any nn.Module torch model.

This module provides a feature extractor implementation that leverages the timm
library to extract intermediate features from various CNN architectures. If a
nn.Module is passed as backbone argument, the TorchFX feature extractor is
used to extract features of the given layers.

Example:
    >>> import torch
    >>> from anomalib.models.components.feature_extractors import (
    ...     TimmFeatureExtractor
    ... )
    >>> # Initialize feature extractor
    >>> extractor = TimmFeatureExtractor(
    ...     backbone="resnet18",
    ...     layers=["layer1", "layer2", "layer3"]
    ... )
    >>> # Extract features from input
    >>> inputs = torch.randn(32, 3, 256, 256)
    >>> features = extractor(inputs)
    >>> # Access features by layer name
    >>> print(features["layer1"].shape)
    torch.Size([32, 64, 64, 64])
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence

import timm
import torch
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor

from .utils import dryrun_find_featuremap_dims

logger = logging.getLogger(__name__)


class TimmFeatureExtractor(nn.Module):
    """Extract intermediate features from timm models or any nn.Module torch model.

    Args:
        backbone (str | nn.Module): Name of the timm model architecture or any torch model to use as backbone.
        layers (Sequence[str]): Names of layers from which to extract features.
        pre_trained (bool, optional): Whether to use pre-trained weights.
            Defaults to ``True``.
        requires_grad (bool, optional): Whether to compute gradients for the
            backbone. Required for training models like STFPM. Defaults to
            ``False``.

    Attributes:
        backbone (str | nn.Module): Name of the backbone model or actual torch backbone model.
        layers (list[str]): Layer names for feature extraction.
        idx (list[int]): Indices mapping layer names to model outputs.
        requires_grad (bool): Whether gradients are computed.
        feature_extractor (nn.Module): The underlying timm model.
        out_dims (list[int]): Output dimensions for each extracted layer.

    Example:
        >>> import torch
        >>> import torchvision
        >>> from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights

        >>> from anomalib.models.components.feature_extractors import (
        ...     TimmFeatureExtractor
        ... )
        >>> # Create extractor
        >>> model = TimmFeatureExtractor(
        ...     backbone="resnet18",
        ...     layers=["layer1", "layer2"]
        ... )
        >>> # Extract features
        >>> inputs = torch.randn(1, 3, 224, 224)
        >>> features = model(inputs)
        >>> # Print shapes
        >>> for name, feat in features.items():
        ...     print(f"{name}: {feat.shape}")
        layer1: torch.Size([1, 64, 56, 56])
        layer2: torch.Size([1, 128, 28, 28])

        >>> # Custom backbone model
        >>> custom_backbone = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
        >>> model = TimmFeatureExtractor(
        ...    backbone=custom_backbone,
        ...    layers=["features.6.8"])
        >>> features = model(inputs)
        >>> # Print shapes
        >>> for name, feat in features.items():
        ...     print(f"{name}: {feat.shape}")
        features.6.8: torch.Size([32, 304, 8, 8])

    """

    def __init__(
        self,
        backbone: str | nn.Module,
        layers: Sequence[str],
        pre_trained: bool = True,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.layers = list(layers)
        self.requires_grad = requires_grad

        if isinstance(backbone, nn.Module):
            self.feature_extractor = create_feature_extractor(
                backbone,
                return_nodes={layer: layer for layer in self.layers},
            )
            layer_metadata = dryrun_find_featuremap_dims(self.feature_extractor, (256, 256), layers=self.layers)
            self.out_dims = [feature_info["num_features"] for layer_name, feature_info in layer_metadata.items()]

        elif isinstance(backbone, str):
            self.idx = self._map_layer_to_idx()
            self.feature_extractor = timm.create_model(
                backbone,
                pretrained=pre_trained,
                pretrained_cfg=None,
                features_only=True,
                exportable=True,
                out_indices=self.idx,
            )
            self.out_dims = self.feature_extractor.feature_info.channels()

        else:
            msg = f"Backbone of type {type(backbone)} must be of type str or nn.Module."
            raise TypeError(msg)

        self._features = {layer: torch.empty(0) for layer in self.layers}

    def _map_layer_to_idx(self) -> list[int]:
        """Map layer names to their indices in the model's output.

        Returns:
            list[int]: Indices corresponding to the requested layer names.

        Note:
            If a requested layer is not found in the model, it is removed from
            ``self.layers`` and a warning is logged.
        """
        idx = []
        model = timm.create_model(
            self.backbone,
            pretrained=False,
            features_only=True,
            exportable=True,
        )
        # model.feature_info.info returns list of dicts containing info,
        # inside which "module" contains layer name
        layer_names = [info["module"] for info in model.feature_info.info]
        for layer in self.layers:
            try:
                idx.append(layer_names.index(layer))
            except ValueError:  # noqa: PERF203
                msg = f"Layer {layer} not found in model {self.backbone}. Available layers: {layer_names}"
                logger.warning(msg)
                # Remove unfound key from layer dict
                self.layers.remove(layer)

        return idx

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract features from the input tensor.

        Args:
            inputs (torch.Tensor): Input tensor of shape
                ``(batch_size, channels, height, width)``.

        Returns:
            dict[str, torch.Tensor]: Dictionary mapping layer names to their
            feature tensors.

        Example:
            >>> import torch
            >>> from anomalib.models.components.feature_extractors import (
            ...     TimmFeatureExtractor
            ... )
            >>> model = TimmFeatureExtractor(
            ...     backbone="resnet18",
            ...     layers=["layer1"]
            ... )
            >>> inputs = torch.randn(1, 3, 224, 224)
            >>> features = model(inputs)
            >>> features["layer1"].shape
            torch.Size([1, 64, 56, 56])
        """
        if self.requires_grad:
            features = self.feature_extractor(inputs)
        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                features = self.feature_extractor(inputs)
        if not isinstance(features, dict):
            features = dict(zip(self.layers, features, strict=True))
        return features
