"""Feature Extractor.

This script extracts features from a CNN network
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Dict, List

import timm
import torch
from torch import Tensor, nn


class FeatureExtractor(nn.Module):
    """Extract features from a CNN.

    Args:
        backbone (nn.Module): The backbone to which the feature extraction hooks are attached.
        layers (Iterable[str]): List of layer names of the backbone to which the hooks are attached.

    Example:
        >>> import torch
        >>> from anomalib.core.model.feature_extractor import FeatureExtractor

        >>> model = FeatureExtractor(model="resnet18", layers=['layer1', 'layer2', 'layer3'])
        >>> input = torch.rand((32, 3, 256, 256))
        >>> features = model(input)

        >>> [layer for layer in features.keys()]
            ['layer1', 'layer2', 'layer3']
        >>> [feature.shape for feature in features.values()]
            [torch.Size([32, 64, 64, 64]), torch.Size([32, 128, 32, 32]), torch.Size([32, 256, 16, 16])]
    """

    def __init__(self, backbone: str, layers: List[str], pre_trained: bool = True):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.idx = self._map_layer_to_idx()
        self.feature_extractor = timm.create_model(
            backbone,
            pretrained=pre_trained,
            features_only=True,
            exportable=True,
            out_indices=self.idx,
        )
        self.out_dims = self.feature_extractor.feature_info.channels()
        self._features = {layer: torch.empty(0) for layer in self.layers}

    def _map_layer_to_idx(self, offset: int = 3) -> List[int]:
        """Maps set of layer names to indices of model.

        Args:
            offset (int) `timm` ignores the first few layers when indexing please update offset based on need

        Returns:
            Feature map extracted from the CNN
        """
        idx = []
        features = timm.create_model(
            self.backbone,
            pretrained=False,
            features_only=False,
            exportable=True,
        )
        for i in self.layers:
            try:
                idx.append(list(dict(features.named_children()).keys()).index(i) - offset)
            except ValueError:
                warnings.warn(f"Layer {i} not found in model {self.backbone}")
                # Remove unfound key from layer dict
                self.layers.remove(i)

        return idx

    def forward(self, input_tensor: Tensor) -> Dict[str, Tensor]:
        """Forward-pass input tensor into the CNN.

        Args:
            input_tensor (Tensor): Input tensor

        Returns:
            Feature map extracted from the CNN
        """
        features = dict(zip(self.layers, self.feature_extractor(input_tensor)))
        return features
