"""Feature extractor using timm models.

This module provides a feature extractor implementation that leverages the timm
library to extract intermediate features from various CNN architectures.

Example:
    >>> import torch
    >>> from anomalib.models.components.feature_extractors import (
    ...     TimmFeatureExtractor
    ... )
    >>> # Initialize feature extractor
    >>> extractor = TimmFeatureExtractor(
    ...     backbone="resnet18",
    ...     layers=["layer1", "layer2", "layer3"]
    ... )
    >>> # Extract features from input
    >>> inputs = torch.randn(32, 3, 256, 256)
    >>> features = extractor(inputs)
    >>> # Access features by layer name
    >>> print(features["layer1"].shape)
    torch.Size([32, 64, 64, 64])
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable, Sequence

import timm
import torch
from torch import nn

logger = logging.getLogger(__name__)


def register_model_with_adv_trained_weights_tags(
    model_name: str,
    epsilons: Sequence[float],
    lp: str,
    cfg_fn: Callable,
) -> None:
    """Register adversarially trained model weights with a URL."""
    from timm.models._registry import _model_pretrained_cfgs as model_pretrained_cfgs
    from timm.models._registry import generate_default_cfgs

    origin_url = "https://huggingface.co/madrylab/robust-imagenet-models"
    paper_ids = "arXiv:2007.08489"

    cfgs = {}
    for eps in epsilons:
        url = f"https://huggingface.co/mzweilin/robust-imagenet-models/resolve/main/{model_name}_{lp}_eps{eps}.pth"
        tag = f"adv_{lp}_{eps}"
        model_and_tag = f"{model_name}.{tag}"
        cfgs[model_and_tag] = cfg_fn(
            url=url,
            origin_url=origin_url,
            paper_ids=paper_ids,
        )

    default_cfgs = generate_default_cfgs(cfgs)

    for model_and_tag in cfgs:
        tag = model_and_tag[len(model_name) + 1 :]  # Remove "[MODEL NAME]."
        if model_and_tag in model_pretrained_cfgs:
            logger.warning(f"Overriding model weights registration in timm: {model_and_tag}")
        model_pretrained_cfgs[model_and_tag] = default_cfgs[model_name].cfgs[tag]
        logger.info(f"Register model weights in timm: {model_and_tag}")


def register_in_bulk() -> None:
    """Register adversarially trained model weights in timm."""
    from timm.models.resnet import _cfg as resnet_cfg_fn

    l2_epsilons = [0, 0.01, 0.03, 0.05, 0.1, 0.25, 0.5, 1, 3, 5]
    linf_epsilons = [0, 0.5, 1, 2, 4, 8]
    model_names = ["resnet18", "resnet50", "wide_resnet50_2"]
    cfg_fn = resnet_cfg_fn
    for model_name in model_names:
        register_model_with_adv_trained_weights_tags(
            model_name=model_name,
            epsilons=l2_epsilons,
            lp="l2",
            cfg_fn=cfg_fn,
        )
        register_model_with_adv_trained_weights_tags(
            model_name=model_name,
            epsilons=linf_epsilons,
            lp="linf",
            cfg_fn=cfg_fn,
        )


def try_register_in_bulk() -> None:
    """Catch the error in case we cannot register new weights in timm due to changes of internal APIs."""
    try:
        register_in_bulk()
    except ImportError as e:
        logger.warning(
            f"Adversarially trained backbones are not available. An error occured when registering weights: {e}",
        )


# We will register model weights only once even if we import the module repeatedly, because it is a singleton.
try_register_in_bulk()


class TimmFeatureExtractor(nn.Module):
    """Extract intermediate features from timm models.

    Args:
        backbone (str): Name of the timm model architecture to use as backbone.
            Can include custom weights URI in format ``name__AT__uri``.
        layers (Sequence[str]): Names of layers from which to extract features.
        pre_trained (bool, optional): Whether to use pre-trained weights.
            Defaults to ``True``.
        requires_grad (bool, optional): Whether to compute gradients for the
            backbone. Required for training models like STFPM. Defaults to
            ``False``.

    Attributes:
        backbone (str): Name of the backbone model.
        layers (list[str]): Layer names for feature extraction.
        idx (list[int]): Indices mapping layer names to model outputs.
        requires_grad (bool): Whether gradients are computed.
        feature_extractor (nn.Module): The underlying timm model.
        out_dims (list[int]): Output dimensions for each extracted layer.

    Example:
        >>> import torch
        >>> from anomalib.models.components.feature_extractors import (
        ...     TimmFeatureExtractor
        ... )
        >>> # Create extractor
        >>> model = TimmFeatureExtractor(
        ...     backbone="resnet18",
        ...     layers=["layer1", "layer2"]
        ... )
        >>> # Extract features
        >>> inputs = torch.randn(1, 3, 224, 224)
        >>> features = model(inputs)
        >>> # Print shapes
        >>> for name, feat in features.items():
        ...     print(f"{name}: {feat.shape}")
        layer1: torch.Size([1, 64, 56, 56])
        layer2: torch.Size([1, 128, 28, 28])
    """

    def __init__(
        self,
        backbone: str,
        layers: Sequence[str],
        pre_trained: bool = True,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.layers = list(layers)
        self.idx = self._map_layer_to_idx()
        self.requires_grad = requires_grad
        self.feature_extractor = timm.create_model(
            backbone,
            pretrained=pre_trained,
            features_only=True,
            exportable=True,
            out_indices=self.idx,
        )
        self.out_dims = self.feature_extractor.feature_info.channels()
        self._features = {layer: torch.empty(0) for layer in self.layers}

    def _map_layer_to_idx(self) -> list[int]:
        """Map layer names to their indices in the model's output.

        Returns:
            list[int]: Indices corresponding to the requested layer names.

        Note:
            If a requested layer is not found in the model, it is removed from
            ``self.layers`` and a warning is logged.
        """
        idx = []
        model = timm.create_model(
            self.backbone,
            pretrained=False,
            features_only=True,
            exportable=True,
        )
        # model.feature_info.info returns list of dicts containing info,
        # inside which "module" contains layer name
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
        """Extract features from the input tensor.

        Args:
            inputs (torch.Tensor): Input tensor of shape
                ``(batch_size, channels, height, width)``.

        Returns:
            dict[str, torch.Tensor]: Dictionary mapping layer names to their
            feature tensors.

        Example:
            >>> import torch
            >>> from anomalib.models.components.feature_extractors import (
            ...     TimmFeatureExtractor
            ... )
            >>> model = TimmFeatureExtractor(
            ...     backbone="resnet18",
            ...     layers=["layer1"]
            ... )
            >>> inputs = torch.randn(1, 3, 224, 224)
            >>> features = model(inputs)
            >>> features["layer1"].shape
            torch.Size([1, 64, 56, 56])
        """
        if self.requires_grad:
            features = dict(zip(self.layers, self.feature_extractor(inputs), strict=True))
        else:
            self.feature_extractor.eval()
            with torch.no_grad():
                features = dict(zip(self.layers, self.feature_extractor(inputs), strict=True))
        return features
