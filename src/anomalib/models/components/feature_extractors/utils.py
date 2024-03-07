"""Utility functions to manipulate feature extractors."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.fx.graph_module import GraphModule

from .timm import TimmFeatureExtractor


def dryrun_find_featuremap_dims(
    feature_extractor: TimmFeatureExtractor | GraphModule,
    input_size: tuple[int, int],
    layers: list[str],
) -> dict[str, dict[str, int | tuple[int, int]]]:
    """Dry run an empty image of `input_size` size to get the featuremap tensors' dimensions (num_features, resolution).

    Returns:
        tuple[int, int]: maping of `layer -> dimensions dict`
            Each `dimension dict` has two keys: `num_features` (int) and `resolution`(tuple[int, int]).
    """
    device = next(feature_extractor.parameters()).device
    dryrun_input = torch.empty(1, 3, *input_size).to(device)
    dryrun_features = feature_extractor(dryrun_input)
    return {
        layer: {"num_features": dryrun_features[layer].shape[1], "resolution": dryrun_features[layer].shape[2:]}
        for layer in layers
    }
