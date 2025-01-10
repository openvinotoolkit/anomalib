"""Feature Extractor based on TorchFX.

This module provides a feature extractor implementation that leverages TorchFX to
extract intermediate features from CNN architectures.

Example:
    >>> import torch
    >>> from anomalib.models.components.feature_extractors import (
    ...     TorchFXFeatureExtractor
    ... )
    >>> # Initialize with torchvision model
    >>> from torchvision.models.efficientnet import EfficientNet_B5_Weights
    >>> extractor = TorchFXFeatureExtractor(
    ...     backbone="efficientnet_b5",
    ...     return_nodes=["features.6.8"],
    ...     weights=EfficientNet_B5_Weights.DEFAULT
    ... )
    >>> # Extract features
    >>> inputs = torch.rand((32, 3, 256, 256))
    >>> features = extractor(inputs)
    >>> print([layer for layer in features.keys()])
    ['features.6.8']
    >>> print([feature.shape for feature in features.values()])
    [torch.Size([32, 304, 8, 8])]

    With custom models:
    >>> # Initialize with custom model
    >>> extractor = TorchFXFeatureExtractor(
    ...     "path.to.CustomModel",
    ...     ["linear_relu_stack.3"],
    ...     weights="path/to/weights.pth"
    ... )
    >>> inputs = torch.randn(1, 1, 28, 28)
    >>> features = extractor(inputs)
    >>> print([layer for layer in features.keys()])
    ['linear_relu_stack.3']

    With model instances:
    >>> # Initialize with model instance
    >>> from timm import create_model
    >>> model = create_model("resnet18", pretrained=True)
    >>> extractor = TorchFXFeatureExtractor(model, ["layer1"])
    >>> inputs = torch.rand((32, 3, 256, 256))
    >>> features = extractor(inputs)
    >>> print([layer for layer in features.keys()])
    ['layer1']
    >>> print([feature.shape for feature in features.values()])
    [torch.Size([32, 64, 64, 64])]
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.fx.graph_module import GraphModule
from torchvision.models._api import WeightsEnum
from torchvision.models.feature_extraction import create_feature_extractor


@dataclass
class BackboneParams:
    """Used for serializing the backbone.

    Args:
        class_path (str | type[nn.Module]): Path to the backbone class or the
            class itself.
        init_args (dict): Dictionary of initialization arguments for the backbone.
            Defaults to empty dict.
    """

    class_path: str | type[nn.Module]
    init_args: dict = field(default_factory=dict)


class TorchFXFeatureExtractor(nn.Module):
    """Extract features from a CNN using TorchFX.

    Args:
        backbone (str | BackboneParams | dict | nn.Module): The backbone to which
            the feature extraction hooks are attached. If a string name is
            provided, the model is loaded from torchvision. Otherwise, the model
            class can be provided and it will try to load the weights from the
            provided weights file. Last, an instance of nn.Module can also be
            passed directly.
        return_nodes (list[str]): List of layer names of the backbone to which
            the hooks are attached. You can find the names of these nodes by
            using ``get_graph_node_names`` function.
        weights (str | WeightsEnum | None): Weights enum to use for the model.
            Torchvision models require ``WeightsEnum``. These enums are defined
            in ``torchvision.models.<model>``. You can pass the weights path for
            custom models. Defaults to ``None``.
        requires_grad (bool): Models like ``stfpm`` use the feature extractor for
            training. In such cases we should set ``requires_grad`` to ``True``.
            Defaults to ``False``.
        tracer_kwargs (dict | None): Dictionary of keyword arguments for
            NodePathTracer (which passes them onto it's parent class
            torch.fx.Tracer). Can be used to allow not tracing through a list of
            problematic modules, by passing a list of ``leaf_modules`` as one of
            the ``tracer_kwargs``. Defaults to ``None``.

    Attributes:
        feature_extractor (GraphModule): The TorchFX feature extractor module.

    Example:
        >>> import torch
        >>> from anomalib.models.components.feature_extractors import (
        ...     TorchFXFeatureExtractor
        ... )
        >>> # Initialize with torchvision model
        >>> extractor = TorchFXFeatureExtractor(
        ...     backbone="resnet18",
        ...     return_nodes=["layer1", "layer2"]
        ... )
        >>> # Extract features
        >>> inputs = torch.randn(1, 3, 224, 224)
        >>> features = extractor(inputs)
        >>> # Access features by layer name
        >>> print(features["layer1"].shape)
        torch.Size([1, 64, 56, 56])
    """

    def __init__(
        self,
        backbone: str | BackboneParams | dict | nn.Module,
        return_nodes: list[str],
        weights: str | WeightsEnum | None = None,
        requires_grad: bool = False,
        tracer_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        if isinstance(backbone, dict):
            backbone = BackboneParams(**backbone)
        elif isinstance(backbone, str):
            backbone = BackboneParams(class_path=backbone)
        elif not isinstance(backbone, nn.Module | BackboneParams):
            msg = f"backbone needs to be of type str | BackboneParams | dict | nn.Module, but was type {type(backbone)}"
            raise TypeError(msg)

        self.feature_extractor = self.initialize_feature_extractor(
            backbone,
            return_nodes,
            weights,
            requires_grad,
            tracer_kwargs,
        )

    def initialize_feature_extractor(
        self,
        backbone: BackboneParams | nn.Module,
        return_nodes: list[str],
        weights: str | WeightsEnum | None = None,
        requires_grad: bool = False,
        tracer_kwargs: dict | None = None,
    ) -> GraphModule:
        """Initialize the feature extractor.

        Args:
            backbone (BackboneParams | nn.Module): The backbone to which the
                feature extraction hooks are attached.
            return_nodes (list[str]): List of layer names to extract features
                from.
            weights (str | WeightsEnum | None): Model weights specification.
                Defaults to ``None``.
            requires_grad (bool): Whether to compute gradients. Defaults to
                ``False``.
            tracer_kwargs (dict | None): Additional arguments for the tracer.
                Defaults to ``None``.

        Returns:
            GraphModule: Initialized feature extractor.

        Raises:
            TypeError: If weights format is invalid.
        """
        if isinstance(backbone, nn.Module):
            backbone_model = backbone
        elif isinstance(backbone.class_path, str):
            backbone_class = self._get_backbone_class(backbone.class_path)
            backbone_model = backbone_class(weights=weights, **backbone.init_args)
        else:
            backbone_class = backbone.class_path
            backbone_model = backbone_class(**backbone.init_args)

        if isinstance(weights, WeightsEnum) or weights in {
            "IMAGENET1K_V1",
            "IMAGENET1K_V2",
            "DEFAULT",
        }:  # torchvision models
            feature_extractor = create_feature_extractor(
                model=backbone_model,
                return_nodes=return_nodes,
            )
        elif weights is not None:
            if not isinstance(weights, str):
                msg = "Weights should point to a path"
                raise TypeError(msg)

            model_weights = torch.load(weights)
            if "state_dict" in model_weights:
                model_weights = model_weights["state_dict"]
            backbone_model.load_state_dict(model_weights)

        feature_extractor = create_feature_extractor(
            backbone_model,
            return_nodes,
            tracer_kwargs=tracer_kwargs,
        )

        if not requires_grad:
            feature_extractor.eval()
            for param in feature_extractor.parameters():
                param.requires_grad_(False)  # noqa: FBT003

        return feature_extractor

    @staticmethod
    def _get_backbone_class(backbone: str) -> Callable[..., nn.Module]:
        """Get the backbone class from the provided path.

        If only the model name is provided, it will try to load the model from
        torchvision.

        Args:
            backbone (str): Path to the backbone class.

        Returns:
            Callable[..., nn.Module]: Backbone class.

        Raises:
            ModuleNotFoundError: If backbone cannot be found.

        Example:
            >>> from anomalib.models.components.feature_extractors import (
            ...     TorchFXFeatureExtractor
            ... )
            >>> # Get torchvision model
            >>> cls = TorchFXFeatureExtractor._get_backbone_class(
            ...     "efficientnet_b5"
            ... )
            >>> # Get custom model
            >>> cls = TorchFXFeatureExtractor._get_backbone_class(
            ...     "path.to.CustomModel"
            ... )
        """
        try:
            if len(backbone.split(".")) > 1:
                # assumes that the entire class path is provided
                models = importlib.import_module(".".join(backbone.split(".")[:-1]))
                backbone_class = getattr(models, backbone.split(".")[-1])
            else:
                models = importlib.import_module("torchvision.models")
                backbone_class = getattr(models, backbone)
        except ModuleNotFoundError as exception:
            msg = f"Backbone {backbone} not found in torchvision.models nor in {backbone} module."
            raise ModuleNotFoundError(msg) from exception

        return backbone_class

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract features from the input.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            dict[str, torch.Tensor]: Dictionary mapping layer names to their
                feature tensors.
        """
        return self.feature_extractor(inputs)
