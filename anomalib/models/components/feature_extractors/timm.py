"""Feature Extractor.

This script extracts features from a CNN network
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import warnings
from typing import Callable, Dict, List, Optional, Union
from timm.models.helpers import load_checkpoint
import timm
import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class TimmFeatureExtractor(nn.Module):
    """Extract features from a CNN.

    Args:
        backbone (nn.Module): The backbone to which the feature extraction hooks are attached.
        layers (Iterable[str]): List of layer names of the backbone to which the hooks are attached.
        pre_trained (bool): Whether to use a pre-trained backbone. Defaults to True.
        requires_grad (bool): Whether to require gradients for the backbone. Defaults to False.
            Models like ``stfpm`` use the feature extractor model as a trainable network. In such cases gradient
            computation is required.
        pretrained_weights (str, optional): Path to pretrained weights. Defaults to None.

    Example:
        >>> import torch
        >>> from anomalib.models.components.feature_extractors import TimmFeatureExtractor

        >>> model = TimmFeatureExtractor(model="resnet18", layers=['layer1', 'layer2', 'layer3'])
        >>> input = torch.rand((32, 3, 256, 256))
        >>> features = model(input)

        >>> [layer for layer in features.keys()]
            ['layer1', 'layer2', 'layer3']
        >>> [feature.shape for feature in features.values()]
            [torch.Size([32, 64, 64, 64]), torch.Size([32, 128, 32, 32]), torch.Size([32, 256, 16, 16])]
    """

    def __init__(
        self,
        backbone: str,
        layers: List[str],
        pre_trained: bool = True,
        requires_grad: bool = False,
        pretrained_weights: Optional[str] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.requires_grad = requires_grad
        self._features = {layer: torch.empty(0) for layer in self.layers}
        if isinstance(self.backbone, str):
            self.modality = "timm"
            self.idx = self._map_layer_to_idx()
            self.feature_extractor = timm.create_model(
                backbone,
                pretrained=pre_trained,
                features_only=True,
                exportable=True,
                out_indices=self.idx,
            )

            if pretrained_weights is not None:
                logger.info("Loading pretrained weights")
                # I'm loading checkpoints here and not from the create model because I want strict=False
                load_checkpoint(self.feature_extractor, pretrained_weights, strict=False)

            self.out_dims = self.feature_extractor.feature_info.channels()
        else:
            if pretrained_weights is not None:
                logger.info("Loading pretrained weights")

                with open(pretrained_weights, "rb") as f:
                    weights = torch.load(f)

                self.backbone.load_state_dict(weights["state_dict"], strict=False)

            self.modality = "module"
            self.out_dims = []
            for layer_id in layers:
                layer = dict([*self.backbone.named_modules()])[layer_id]
                layer.register_forward_hook(self.get_features(layer_id))
                # get output dimension of features if available
                layer_modules = [*layer.modules()]
                for idx in reversed(range(len(layer_modules))):
                    if hasattr(layer_modules[idx], "out_channels"):
                        self.out_dims.append(layer_modules[idx].out_channels)
                        break

    def _map_layer_to_idx(self, offset: int = 3) -> list[int]:
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

    def get_features(self, layer_id: str) -> Callable:
        """Get layer features.

        Args:
            layer_id (str): Layer ID

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

    def forward(self, inputs: Tensor) -> Dict[str, Tensor]:
        """Forward-pass input tensor into the CNN.

        Args:
            inputs (Tensor): Input tensor

        Returns:
            Feature map extracted from the CNN
        """
        if self.modality == "timm":
            if self.requires_grad:
                features = dict(zip(self.layers, self.feature_extractor(inputs)))
            else:
                self.feature_extractor.eval()
                with torch.no_grad():
                    features = dict(zip(self.layers, self.feature_extractor(inputs)))

            return features
        else:
            self._features = {layer: torch.empty(0) for layer in self.layers}
            _ = self.backbone(inputs)
            return self._features


class FeatureExtractor(TimmFeatureExtractor):
    """Compatibility wrapper for the old FeatureExtractor class.

    See :class:`anomalib.models.components.feature_extractors.timm.TimmFeatureExtractor` for more details.
    """

    def __init__(self, *args, **kwargs):
        logger.warning(
            "FeatureExtractor is deprecated. Use TimmFeatureExtractor instead."
            " Both FeatureExtractor and TimmFeatureExtractor will be removed in a future release."
        )
        super().__init__(*args, **kwargs)
