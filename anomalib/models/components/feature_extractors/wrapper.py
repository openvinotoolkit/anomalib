"""Convenience wrapper for feature extraction methods."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Union

from omegaconf import DictConfig

from .timm import TimmFeatureExtractor, TimmFeatureExtractorParams
from .torchfx import TorchFXFeatureExtractor, TorchFXFeatureExtractorParams

FeatureExtractorParams = Union[TimmFeatureExtractorParams, TorchFXFeatureExtractorParams]


def get_feature_extractor(
    *args: Union[TimmFeatureExtractorParams, TorchFXFeatureExtractorParams, DictConfig], **kwargs
) -> Union[TorchFXFeatureExtractor, TimmFeatureExtractor]:
    """Convenience wrapper for feature extractors.

    Selects either timm or torchfx feature extractor based on the arguments passed.

    If you want to use timm feature extractor, you need to pass the following arguments:
    backbone,layers,pre_trained,requires_grad

    If you want to use torchfx feature extractor, you need to pass the following arguments:
    backbone,return_nodes,weights,requires_grad

    Example:
        Using Timm

        >>> from anomalib.models.components import FeatureExtractor
        >>> FeatureExtractor(backbone="resnet18",layers=["layer1","layer2","layer3"])
        TimmFeatureExtractor will be removed in 2023.1
        FeatureExtractor(
        (feature_extractor): TimmFeatureExtractor(
            (feature_extractor): FeatureListNet(
                ...

        Using TorchFX

        >>> FeatureExtractor(backbone="resnet18",return_nodes=["layer1","layer2","layer3"])
            FeatureExtractor(
            (feature_extractor): TorchFXFeatureExtractor(
                (feature_extractor): ResNet(

        Using Backbone params

        >>> from anomalib.models.components.feature_extractors import TorchFXFeatureExtractorParams
        >>> from torchvision.models.efficientnet import EfficientNet_B5_Weights
        >>> params = TorchFXFeatureExtractorParams(backbone="efficientnet_b5",
        ...                     return_nodes=["features.6.8"],
        ...                     weights=EfficientNet_B5_Weights.DEFAULT
        ... )
        >>> FeatureExtractor(params)
        FeatureExtractor(
        (feature_extractor): TorchFXFeatureExtractor(
            (feature_extractor): EfficientNet(
                ...
    """
    feature_extractor_params = _get_feature_extractor_params(args, kwargs)
    if isinstance(feature_extractor_params, TimmFeatureExtractorParams):
        feature_extractor = TimmFeatureExtractor(**vars(feature_extractor_params))
    else:
        feature_extractor = TorchFXFeatureExtractor(**vars(feature_extractor_params))
    return feature_extractor


def _get_feature_extractor_params(args, kwargs):
    """Performs validation checks and converts the arguments to the correct data type.

    Checks if the arguments are passed as a key word argument or as a single argument of dictionary or dataclass.
    If the checks pass, returns the feature extractor parameters as a dataclass.

    The feature extractor expects only one of args of kwargs

    Args:
        args (Union[TimmFeatureExtractorParams, TorchFXFeatureExtractorParams, DictConfig]): Feature extractor
            parameters.
        kwargs (Dict[str, Any]): Feature extractor parameters as key word arguments.
    """
    if len(args) == 1:
        feature_extractor_params = _convert_datatype(args[0])
    elif len(args) > 0 and kwargs is not None:
        raise ValueError(
            "Either arguments as keyword arguments or as a single argument of type TimmFeatureExtractorParams or"
            " TorchFXFeatureExtractorParams"
        )
    else:
        feature_extractor_params = _convert_datatype(kwargs)
    return feature_extractor_params


def _convert_datatype(
    feature_extractor_params: Union[TimmFeatureExtractorParams, TorchFXFeatureExtractorParams, DictConfig, Dict],
) -> FeatureExtractorParams:
    """When config us loaded from entry point scripts, the data type of the arguments is DictConfig.

    Args:
        feature_extractor_params: Feature extractor parameters to convert.

    Returns:
        FeatureExtractorParams: Converted feature extractor parameters.
    """
    if isinstance(feature_extractor_params, (DictConfig, dict)):
        if "layers" in feature_extractor_params:
            feature_extractor_params = TimmFeatureExtractorParams(**feature_extractor_params)
        else:
            feature_extractor_params = TorchFXFeatureExtractorParams(**feature_extractor_params)
    if not isinstance(feature_extractor_params, (TimmFeatureExtractorParams, TorchFXFeatureExtractorParams)):
        raise ValueError(f"Unknown feature extractor params type: {type(feature_extractor_params)}")
    return feature_extractor_params
