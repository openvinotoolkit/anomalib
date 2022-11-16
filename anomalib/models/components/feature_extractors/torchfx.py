"""Feature Extractor based on TorchFX."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import List, Optional, Union

import torch
from torch import nn
from torch.fx.graph_module import GraphModule
from torchvision.models._api import WeightsEnum
from torchvision.models.feature_extraction import create_feature_extractor


def get_torchfx_feature_extractor(
    backbone: Union[str, nn.Module],
    return_nodes: List[str],
    weights: Optional[Union[WeightsEnum, str]] = None,
    requires_grad: bool = False,
    **kwargs,
) -> GraphModule:
    """Extract features from a CNN.

    Args:
        backbone (Union[str, nn.Module]): The backbone to which the feature extraction hooks are attached.
            If the name is provided, the model is loaded from torchvision. Otherwise, the model class can be provided
            and it will try to load the weights from the provided weights file.
        return_nodes (Iterable[str]): List of layer names of the backbone to which the hooks are attached.
            You can find the names of these nodes by using ``get_graph_node_names`` function.
        weights (Optional[Union[WeightsEnum,str]]): Weights enum to use for the model. Torchvision models require
            ``WeightsEnum``. These enums are defined in ``torchvision.models.<model>``. You can pass the weights path
            for custom models.
        requires_grad (bool): Models like ``stfpm`` use the feature extractor for training. In such cases we should
            set ``requires_grad`` to ``True``. Default is ``False``.
        **kwargs: Additional arguments to pass to the model class.

    Example:
        With torchvision models:

            >>> import torch
            >>> from anomalib.models.components.feature_extractors import get_torchfx_feature_extractor
            >>> from torchvision.models.efficientnet import EfficientNet_B5_Weights
            >>> feature_extractor = get_torchfx_feature_extractor(
                    backbone="efficientnet_b5", return_nodes=["features.6.8"], weights=EfficientNet_B5_Weights.DEFAULT
                )
            >>> input = torch.rand((32, 3, 256, 256))
            >>> features = feature_extractor(input)
            >>> [layer for layer in features.keys()]
                ["features.6.8"]
            >>> [feature.shape for feature in features.values()]
                [torch.Size([32, 304, 8, 8])]

        With custom models:

            >>> import CustomModel
            >>> from anomalib.models.components.feature_extractors import get_torchfx_feature_extractor
            >>> feature_extractor = get_torchfx_feature_extractor(
                    CustomModel, ["linear_relu_stack.3"], weights="path/to/weights.pth"
                )
            >>> input = torch.randn(1, 1, 28, 28)
            >>> features = feature_extractor(input)
            >>> [layer for layer in features.keys()]
                ["linear_relu_stack.3"]

    Returns:
        Feature Extractor based on TorchFX.
    """
    # Get torchvision feature extractor
    if isinstance(backbone, str):
        try:
            models = importlib.import_module("torchvision.models")
            backbone_model = getattr(models, backbone)
        except ModuleNotFoundError as exception:
            raise ModuleNotFoundError(f"Backbone {backbone} not found in torchvision.models") from exception
        if weights is not None:
            assert isinstance(weights, WeightsEnum), "Weights should be of type WeightsEnum"
        feature_extractor = create_feature_extractor(backbone_model(weights=weights), return_nodes)
    # Load model from ``nn.Module`` class
    else:
        backbone_model = backbone(**kwargs)
        if weights is not None:
            assert isinstance(weights, str), "Weights should point to a path"
            backbone_model.load_state_dict(torch.load(weights)["state_dict"])
        feature_extractor = create_feature_extractor(backbone_model, return_nodes)

    if not requires_grad:
        feature_extractor.eval()
        for param in feature_extractor.parameters():
            param.requires_grad_(False)

    return feature_extractor
