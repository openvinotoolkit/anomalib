"""Feature extractor using timm models.

This module provides a feature extractor implementation that leverages the timm
library to extract intermediate features from various CNN architectures.

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

logger = logging.getLogger(__name__)


class TimmFeatureExtractor(nn.Module):
    """Extract intermediate features from timm models.

    Args:
        backbone (str): Name of the timm model architecture to use as backbone.
            Can include custom weights URI in format ``name__AT__uri``.
        layers (Sequence[str]): Names of layers from which to extract features.
        pre_trained (bool, optional): Whether to use pre-trained weights.
            Defaults to ``True``.
        requires_grad (bool, optional): Whether to compute gradients for the
            backbone. Required for training models like STFPM. Defaults to
            ``False``.

    Attributes:
        backbone (str): Name of the backbone model.
        layers (list[str]): Layer names for feature extraction.
        idx (list[int]): Indices mapping layer names to model outputs.
        requires_grad (bool): Whether gradients are computed.
        feature_extractor (nn.Module): The underlying timm model.
        out_dims (list[int]): Output dimensions for each extracted layer.

    Example:
        >>> import torch
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
    """

    def __init__(
        self,
        backbone: str,
        layers: Sequence[str],
        pre_trained: bool = True,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()

        # Extract backbone-name and weight-URI from the backbone string.
        if "__AT__" in backbone:
            backbone, uri = backbone.split("__AT__")
            pretrained_cfg = timm.models.registry.get_pretrained_cfg(backbone)
            # Override pretrained_cfg["url"] to use different pretrained weights.
            pretrained_cfg["url"] = uri
        else:
            pretrained_cfg = None

        self.backbone = backbone
        self.layers = list(layers)
        self.idx = self._map_layer_to_idx()
        self.requires_grad = requires_grad
        self.feature_extractor = timm.create_model(
            backbone,
            pretrained=pre_trained,
            pretrained_cfg=pretrained_cfg,
            features_only=True,
            exportable=True,
            out_indices=self.idx,
        )
        self.out_dims = self.feature_extractor.feature_info.channels()
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
            features = dict(zip(self.layers, self.feature_extractor(inputs), strict=True))
        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                features = dict(zip(self.layers, self.feature_extractor(inputs), strict=True))
        return features
