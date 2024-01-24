"""Feature Extractor.

This script extracts features from a CNN network
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Sequence

import timm
import torch
from torch import nn

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

    Example:
        .. code-block:: python

            import torch
            from anomalib.models.components.feature_extractors import TimmFeatureExtractor

            model = TimmFeatureExtractor(model="resnet18", layers=['layer1', 'layer2', 'layer3'])
            input = torch.rand((32, 3, 256, 256))
            features = model(input)

            print([layer for layer in features.keys()])
            # Output: ['layer1', 'layer2', 'layer3']

            print([feature.shape for feature in features.values()]()
            # Output: [torch.Size([32, 64, 64, 64]), torch.Size([32, 128, 32, 32]), torch.Size([32, 256, 16, 16])]
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
        """Map set of layer names to indices of model.

        Returns:
            list[int]: Feature map extracted from the CNN.
        """
        idx = []
        model = timm.create_model(
            self.backbone,
            pretrained=False,
            features_only=True,
            exportable=True,
        )
        # model.feature_info.info returns list of dicts containing info, inside which "module" contains layer name
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
        """Forward-pass input tensor into the CNN.

        Args:
            inputs (torch.Tensor): Input tensor

        Returns:
            Feature map extracted from the CNN

        Example:
            .. code-block:: python

                model = TimmFeatureExtractor(model="resnet50", layers=['layer3'])
                input = torch.rand((32, 3, 256, 256))
                features = model.forward(input)

        """
        if self.requires_grad:
            features = dict(zip(self.layers, self.feature_extractor(inputs), strict=True))
        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                features = dict(zip(self.layers, self.feature_extractor(inputs), strict=True))
        return features
