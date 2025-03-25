"""Feature extractor for Fuvas model."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torchvision.models.feature_extraction import create_feature_extractor


class FeatureExtractor(nn.Module):
    """Feature extractor for deep neural networks.

    Extracts features from specific layers of a pre-trained model.

    Args:
        model: Pre-trained model to extract features from
        layer_name: Name of the layer to extract features from
        pool_factor: Optional pooling factor configuration
    """

    def __init__(self, model: nn.Module, layer_name: str, pool_factor: dict[str, int] | None = None) -> None:
        super().__init__()
        return_nodes: list[str] = [layer_name]
        self.model = create_feature_extractor(model, return_nodes=return_nodes)

        self.layer_name = layer_name
        self.pool_factor = pool_factor
        self.feature_shapes: dict[str, tuple] = {}
        self.feature_shapes_flatten: dict[str, int] = {}

    def __repr__(self) -> str:
        """String representation of the feature extractor."""
        out = f"Layer {self.layer_name}\n"
        if self.pool_factor:
            out = f"{out}Pool factors {list(self.pool_factor.values())}\n"
        return f"{self.model.__repr__()}"

    def forward(self, data: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through the feature extractor.

        Args:
            data: Input data to extract features from.

        Returns:
            Dictionary mapping layer names to extracted features.
        """
        return self.model(data)

    def get_feature_shapes(self) -> tuple:
        """Get the shape of features for the specified layer.

        Returns:
            Tuple representing the shape of the features.
        """
        return self.feature_shapes[self.layer_name]

    def get_flat_shapes(self) -> int:
        """Get the flattened shape dimension for the specified layer.

        Returns:
            Integer representing the flattened dimension.
        """
        return self.feature_shapes_flatten[self.layer_name]
