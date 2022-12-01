"""Convenience wrapper for feature extraction methods."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

import torch
from omegaconf import DictConfig
from torch import Tensor, nn

from .timm import TimmFeatureExtractor, TimmFeatureExtractorParams
from .torchfx import TorchFXFeatureExtractor, TorchFXFeatureExtractorParams


class FeatureExtractor(nn.Module):
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

    def __init__(self, *args: Union[TimmFeatureExtractorParams, TorchFXFeatureExtractorParams, DictConfig], **kwargs):
        super().__init__()

        # Check if argument is passed as a key word argument or as a single argument of dictionary or dataclass.
        feature_extractor_params = self._get_feature_extractor_params(args, kwargs)
        self.feature_extractor = self._assign_feature_extractor(feature_extractor_params)
        self.layers = (
            self.feature_extractor.layers
            if isinstance(self.feature_extractor, TimmFeatureExtractor)
            else self.feature_extractor.return_nodes
        )
        self._out_dims: List[int]

    def _get_feature_extractor_params(self, args, kwargs):
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
            feature_extractor_params = self._convert_datatype(args[0])
        elif len(args) > 0 and kwargs is not None:
            raise ValueError(
                "Either arguments as keyword arguments or as a single argument of type TimmFeatureExtractorParams or"
                " TorchFXFeatureExtractorParams"
            )
        else:
            feature_extractor_params = self._convert_datatype(kwargs)
        return feature_extractor_params

    def _convert_datatype(
        self,
        feature_extractor_params: Union[TimmFeatureExtractorParams, TorchFXFeatureExtractorParams, DictConfig, Dict],
    ):
        """When config us loaded from entry point scripts, the data type of the arguments is DictConfig.

        Args:
            feature_extractor_params: Feature extractor parameters to convert.

        Returns:
            Union[TimmFeatureExtractorParams, TorchFXFeatureExtractorParams]: Converted feature extractor parameters.
        """
        if isinstance(feature_extractor_params, (DictConfig, dict)):
            if "layers" in feature_extractor_params:
                feature_extractor_params = TimmFeatureExtractorParams(**feature_extractor_params)
            else:
                feature_extractor_params = TorchFXFeatureExtractorParams(**feature_extractor_params)
        if not isinstance(feature_extractor_params, (TimmFeatureExtractorParams, TorchFXFeatureExtractorParams)):
            raise ValueError(f"Unknown feature extractor params type: {type(feature_extractor_params)}")
        return feature_extractor_params

    def _assign_feature_extractor(
        self, feature_extractor_params: Union[TimmFeatureExtractorParams, TorchFXFeatureExtractorParams]
    ) -> Union[TimmFeatureExtractor, TorchFXFeatureExtractor]:
        """Assigns the feature extractor based on the arguments passed."""
        if isinstance(feature_extractor_params, TimmFeatureExtractorParams):
            feature_extractor = TimmFeatureExtractor(**vars(feature_extractor_params))
        else:
            feature_extractor = TorchFXFeatureExtractor(**vars(feature_extractor_params))
        return feature_extractor

    def forward(self, inputs: Tensor) -> Tensor:
        """Returns the feature maps from the selected feature extractor."""
        return self.feature_extractor(inputs)

    @property
    def out_dims(self) -> List[int]:
        """Returns the number of channels of the requested layers."""
        if not hasattr(self, "_out_dims"):
            if isinstance(self.feature_extractor, TimmFeatureExtractor):
                self._out_dims = self.feature_extractor.out_dims
            else:
                # run a small tensor through the model to get the output dimensions
                self._out_dims = [val["num_features"] for val in self.dryrun_find_featuremap_dims((1, 1)).values()]
        return self._out_dims

    def dryrun_find_featuremap_dims(self, input_shape: Tuple[int, int]) -> Dict[str, Dict]:
        """Dry run an empty image of get the feature map tensors' dimensions (num_features, resolution).

        Args:
            input_shape (Tuple[int, int]): Shape of the input image.

        Returns:
            Dict[str, Tuple]: mapping of ```layer -> dimensions dict```
                Each `dimension dict` has two keys: `num_features` (int) and ```resolution```(Tuple[int, int]).
        """

        dryrun_input = torch.empty(1, 3, *input_shape)
        dryrun_features = self.feature_extractor(dryrun_input)
        return {
            layer: {"num_features": dryrun_features[layer].shape[1], "resolution": dryrun_features[layer].shape[2:]}
            for layer in self.layers
        }
