"""Feature Extractor.

This script extracts features from a CNN network
"""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Callable, Dict, Iterable

import torch
from torch import Tensor, nn


class FeatureExtractor(nn.Module):
    """Extract features from a CNN.

    Example:
        >>> import torch
        >>> import torchvision
        >>> from anomalib.core.model.feature_extractor import FeatureExtractor

        >>> model = FeatureExtractor(model=torchvision.models.resnet18(), layers=['layer1', 'layer2', 'layer3'])
        >>> input = torch.rand((32, 3, 256, 256))
        >>> features = model(input)

        >>> [layer for layer in features.keys()]
            ['layer1', 'layer2', 'layer3']
        >>> [feature.shape for feature in features.values()]
            [torch.Size([32, 64, 64, 64]), torch.Size([32, 128, 32, 32]), torch.Size([32, 256, 16, 16])]
    """

    def __init__(self, backbone: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in self.layers}

        for layer_id in layers:
            layer = dict([*self.backbone.named_modules()])[layer_id]
            layer.register_forward_hook(self.get_features(layer_id))

    def get_features(self, layer_id: str) -> Callable:
        """Get layer features.

        Args:
            layer_id: str: Layer ID

        Returns:
            Layer features
        """

        def hook(_, __, output):
            """Hook to extract features via a forward-pass.

            Args:
              output: Feature map collected after the forward-pass.
            """
            self._features[layer_id] = output

        return hook

    def forward(self, input_tensor: Tensor) -> Dict[str, Tensor]:
        """Forward-pass input tensor into the CNN.

        Args:
            input_tensor (Tensor): Input tensor

        Returns:
            Feature map extracted from the CNN
        """
        self._features = {layer: torch.empty(0) for layer in self.layers}
        _ = self.backbone(input_tensor)
        return self._features
