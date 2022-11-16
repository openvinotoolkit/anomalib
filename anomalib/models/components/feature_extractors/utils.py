"""Utility functions to manipulate feature extractors."""

from typing import Dict, List, Tuple, Union

import torch

from anomalib.models.components.feature_extractors.feature_extractor import (
    FeatureExtractor,
)


def dryrun_find_featuremap_dims(
    feature_extractor: FeatureExtractor,
    input_size: Tuple[int, int],
    layers: List[str],
) -> Dict[str, Dict[str, Union[int, Tuple[int, int]]]]:
    """Dry run an empty image of `input_size` size to get the featuremap tensors' dimensions (num_features, resolution).

    Returns:
        Tuple[int, int]: maping of `layer -> dimensions dict`
            Each `dimension dict` has two keys: `num_features` (int) and `resolution`(Tuple[int, int]).
    """

    dryrun_input = torch.empty(1, 3, *input_size)
    dryrun_features = feature_extractor(dryrun_input)
    return {
        layer: {"num_features": dryrun_features[layer].shape[1], "resolution": dryrun_features[layer].shape[2:]}
        for layer in layers
    }
