"""Feature Extractor based on TorchFX."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor, nn
from torch.fx.graph_module import GraphModule
from torchvision.models._api import WeightsEnum
from torchvision.models.feature_extraction import create_feature_extractor


@dataclass
class BackboneParams:
    """Used for serializing the backbone."""

    class_path: str
    init_args: Dict


class TorchFXFeatureExtractor:
    """Extract features from a CNN.

    Args:
        backbone (Union[str, BackboneParams, Dict]): The backbone to which the feature extraction hooks are attached.
            If the name is provided, the model is loaded from torchvision. Otherwise, the model class can be
            provided and it will try to load the weights from the provided weights file.
        return_nodes (Iterable[str]): List of layer names of the backbone to which the hooks are attached.
            You can find the names of these nodes by using ``get_graph_node_names`` function.
        weights (Optional[Union[WeightsEnum,str]]): Weights enum to use for the model. Torchvision models require
            ``WeightsEnum``. These enums are defined in ``torchvision.models.<model>``. You can pass the weights
            path for custom models.
        requires_grad (bool): Models like ``stfpm`` use the feature extractor for training. In such cases we should
            set ``requires_grad`` to ``True``. Default is ``False``.

    Example:
        With torchvision models:

            >>> import torch
            >>> from anomalib.models.components.feature_extractors import TorchFXFeatureExtractor
            >>> from torchvision.models.efficientnet import EfficientNet_B5_Weights
            >>> feature_extractor = TorchFXFeatureExtractor(
                    backbone="efficientnet_b5",
                    return_nodes=["features.6.8"],
                    weights=EfficientNet_B5_Weights.DEFAULT
                )
            >>> input = torch.rand((32, 3, 256, 256))
            >>> features = feature_extractor(input)
            >>> [layer for layer in features.keys()]
                ["features.6.8"]
            >>> [feature.shape for feature in features.values()]
                [torch.Size([32, 304, 8, 8])]

        With custom models:

            >>> from anomalib.models.components.feature_extractors import TorchFXFeatureExtractor
            >>> feature_extractor = TorchFXFeatureExtractor(
                    "path.to.CustomModel", ["linear_relu_stack.3"], weights="path/to/weights.pth"
                )
            >>> input = torch.randn(1, 1, 28, 28)
            >>> features = feature_extractor(input)
            >>> [layer for layer in features.keys()]
                ["linear_relu_stack.3"]
    """

    def __init__(
        self,
        backbone: Union[str, BackboneParams, Dict],
        return_nodes: List[str],
        weights: Optional[Union[WeightsEnum, str]] = None,
        requires_grad: bool = False,
    ):
        if isinstance(backbone, dict):
            backbone = BackboneParams(**backbone)
        elif isinstance(backbone, str):
            backbone = BackboneParams(class_path=backbone, init_args={})

        self.feature_extractor = self.initialize_feature_extractor(backbone, return_nodes, weights, requires_grad)

    def initialize_feature_extractor(
        self,
        backbone: BackboneParams,
        return_nodes: List[str],
        weights: Optional[Union[WeightsEnum, str]] = None,
        requires_grad: bool = False,
    ) -> Union[GraphModule, nn.Module]:
        """Extract features from a CNN.

        Args:
            backbone (Union[str, BackboneParams]): The backbone to which the feature extraction hooks are attached.
                If the name is provided, the model is loaded from torchvision. Otherwise, the model class can be
                provided and it will try to load the weights from the provided weights file.
            return_nodes (Iterable[str]): List of layer names of the backbone to which the hooks are attached.
                You can find the names of these nodes by using ``get_graph_node_names`` function.
            weights (Optional[Union[WeightsEnum,str]]): Weights enum to use for the model. Torchvision models require
                ``WeightsEnum``. These enums are defined in ``torchvision.models.<model>``. You can pass the weights
                path for custom models.
            requires_grad (bool): Models like ``stfpm`` use the feature extractor for training. In such cases we should
                set ``requires_grad`` to ``True``. Default is ``False``.

        Returns:
            Feature Extractor based on TorchFX.
        """
        backbone_model = self.get_backbone_class(backbone.class_path)
        if isinstance(weights, WeightsEnum):  # torchvision models
            feature_extractor = create_feature_extractor(
                backbone_model(weights=weights, **backbone.init_args), return_nodes
            )
        else:
            backbone_model = backbone_model(**backbone.init_args)
            if weights is not None:
                assert isinstance(weights, str), "Weights should point to a path"
                backbone_model.load_state_dict(torch.load(weights)["state_dict"])
            feature_extractor = create_feature_extractor(backbone_model, return_nodes)

        if not requires_grad:
            feature_extractor.eval()
            for param in feature_extractor.parameters():
                param.requires_grad_(False)

        return feature_extractor

    def get_backbone_class(self, backbone: str) -> nn.Module:
        """Get the backbone class from the provided path.

        If only the moodel name is provided, it will try to load the model from torchvision.

        Args:
            backbone (str): Path to the backbone class.

        Returns:
            Backbone class.
        """
        try:
            if len(backbone.split(".")) > 1:
                # assumes that the entire class path is provided
                models = importlib.import_module(".".join(backbone.split(".")[:-1]))
                backbone_model = getattr(models, backbone.split(".")[-1])
            else:
                models = importlib.import_module("torchvision.models")
                backbone_model = getattr(models, backbone)
        except ModuleNotFoundError as exception:
            raise ModuleNotFoundError(
                f"Backbone {backbone} not found in torchvision.models and not found in {backbone} module either"
            ) from exception

        return backbone_model

    def __call__(self, inputs: Tensor) -> Dict[str, Tensor]:
        """Extract features from the input."""
        return self.feature_extractor(inputs)
