"""Utility functions to manipulate feature extractors.

This module provides utility functions for working with feature extractors,
including functions to analyze feature map dimensions.

Example:
    >>> import torch
    >>> from anomalib.models.components.feature_extractors import (
    ...     TimmFeatureExtractor,
    ...     dryrun_find_featuremap_dims
    ... )
    >>> # Create feature extractor
    >>> extractor = TimmFeatureExtractor(
    ...     backbone="resnet18",
    ...     layers=["layer1", "layer2"]
    ... )
    >>> # Get feature dimensions
    >>> dims = dryrun_find_featuremap_dims(
    ...     extractor,
    ...     input_size=(256, 256),
    ...     layers=["layer1", "layer2"]
    ... )
    >>> print(dims["layer1"]["num_features"])  # Number of channels
    64
    >>> print(dims["layer1"]["resolution"])  # Feature map height, width
    (64, 64)
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.fx.graph_module import GraphModule

from .timm import TimmFeatureExtractor


def dryrun_find_featuremap_dims(
    feature_extractor: TimmFeatureExtractor | GraphModule,
    input_size: tuple[int, int],
    layers: list[str],
) -> dict[str, dict[str, int | tuple[int, int]]]:
    """Get feature map dimensions by running an empty tensor through the model.

    Performs a forward pass with an empty tensor to determine the output
    dimensions of specified feature maps.

    Args:
        feature_extractor: Feature extraction model, either a ``TimmFeatureExtractor``
            or ``GraphModule``.
        input_size: Tuple of ``(height, width)`` specifying input image dimensions.
        layers: List of layer names from which to extract features.

    Returns:
        Dictionary mapping layer names to dimension information. For each layer,
        returns a dictionary with:
            - ``num_features``: Number of feature channels (int)
            - ``resolution``: Spatial dimensions as ``(height, width)`` tuple

    Example:
        >>> extractor = TimmFeatureExtractor("resnet18", layers=["layer1"])
        >>> dims = dryrun_find_featuremap_dims(
        ...     extractor,
        ...     input_size=(256, 256),
        ...     layers=["layer1"]
        ... )
        >>> print(dims["layer1"]["num_features"])  # channels
        64
        >>> print(dims["layer1"]["resolution"])  # (height, width)
        (64, 64)
    """
    device = next(feature_extractor.parameters()).device
    dryrun_input = torch.empty(1, 3, *input_size).to(device)
    dryrun_features = feature_extractor(dryrun_input)
    return {
        layer: {
            "num_features": dryrun_features[layer].shape[1],
            "resolution": dryrun_features[layer].shape[2:],
        }
        for layer in layers
    }
