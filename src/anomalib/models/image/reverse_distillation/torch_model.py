"""PyTorch model implementation for Reverse Distillation.

This module implements the core PyTorch model architecture for the Reverse Distillation
anomaly detection method as described in `Deng et al. (2022)
<https://arxiv.org/abs/2201.10703v2>`_.

The model consists of:
- A pre-trained encoder (e.g. ResNet) that extracts multi-scale features
- A bottleneck layer that compresses features into a compact representation
- A decoder that reconstructs features back to the original feature space
- A scoring mechanism based on reconstruction error

Example:
    >>> from anomalib.models.image.reverse_distillation.torch_model import (
    ...     ReverseDistillationModel
    ... )
    >>> model = ReverseDistillationModel(
    ...     backbone="wide_resnet50_2",
    ...     input_size=(256, 256),
    ...     layers=["layer1", "layer2", "layer3"],
    ...     anomaly_map_mode="multiply"
    ... )
    >>> features = model(torch.randn(1, 3, 256, 256))

See Also:
    - :class:`ReverseDistillationModel`: Main PyTorch model implementation
    - :class:`ReverseDistillationLoss`: Loss function for training
    - :class:`AnomalyMapGenerator`: Anomaly map generation from features
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import nn

from anomalib.data import InferenceBatch
from anomalib.models.components import TimmFeatureExtractor

from .anomaly_map import AnomalyMapGenerationMode, AnomalyMapGenerator
from .components import get_bottleneck_layer, get_decoder

if TYPE_CHECKING:
    from anomalib.data.utils.tiler import Tiler


class ReverseDistillationModel(nn.Module):
    """PyTorch implementation of the Reverse Distillation model.

    The model consists of an encoder-decoder architecture where the encoder extracts
    multi-scale features and the decoder reconstructs them back to the original
    feature space. The reconstruction error is used to detect anomalies.

    Args:
        backbone (str): Name of the backbone CNN architecture used for encoder and
            decoder. Supported backbones can be found in timm library.
        input_size (tuple[int, int]): Size of input images in format ``(H, W)``.
        layers (Sequence[str]): Names of layers from which to extract features.
            For example ``["layer1", "layer2", "layer3"]``.
        anomaly_map_mode (AnomalyMapGenerationMode): Mode used to generate anomaly
            map. Options are ``"multiply"`` or ``"add"``.
        pre_trained (bool, optional): Whether to use pre-trained weights for the
            encoder backbone. Defaults to ``True``.

    Example:
        >>> import torch
        >>> from anomalib.models.image.reverse_distillation.torch_model import (
        ...     ReverseDistillationModel
        ... )
        >>> model = ReverseDistillationModel(
        ...     backbone="wide_resnet50_2",
        ...     input_size=(256, 256),
        ...     layers=["layer1", "layer2", "layer3"],
        ...     anomaly_map_mode="multiply"
        ... )
        >>> input_tensor = torch.randn(1, 3, 256, 256)
        >>> features = model(input_tensor)

    Note:
        The original paper uses torchvision's pre-trained wide_resnet50_2 as the
        encoder backbone.

    Attributes:
        tiler (Tiler | None): Optional tiler for processing large images in patches.
        encoder (TimmFeatureExtractor): Feature extraction backbone.
        bottleneck (nn.Module): Bottleneck layer to compress features.
        decoder (nn.Module): Decoder network to reconstruct features.
        anomaly_map_generator (AnomalyMapGenerator): Module to generate anomaly
            maps from features.
    """

    def __init__(
        self,
        backbone: str,
        input_size: tuple[int, int],
        layers: Sequence[str],
        anomaly_map_mode: AnomalyMapGenerationMode,
        pre_trained: bool = True,
    ) -> None:
        super().__init__()
        self.tiler: Tiler | None = None

        encoder_backbone = backbone
        self.encoder = TimmFeatureExtractor(backbone=encoder_backbone, pre_trained=pre_trained, layers=layers)
        self.bottleneck = get_bottleneck_layer(backbone)
        self.decoder = get_decoder(backbone)

        self.anomaly_map_generator = AnomalyMapGenerator(image_size=input_size, mode=anomaly_map_mode)

    def forward(self, images: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]] | InferenceBatch:
        """Forward pass through the model.

        The behavior differs between training and evaluation modes:
        - Training: Returns encoder and decoder features for computing loss
        - Evaluation: Returns anomaly maps and scores

        Args:
            images (torch.Tensor): Input tensor of shape ``(N, C, H, W)`` where
                ``N`` is batch size, ``C`` is number of channels, ``H`` and ``W``
                are height and width.

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]] | InferenceBatch:
                - In training mode: Tuple of lists containing encoder and decoder
                  features
                - In evaluation mode: ``InferenceBatch`` containing anomaly maps
                  and scores

        Example:
            >>> import torch
            >>> model = ReverseDistillationModel(
            ...     backbone="wide_resnet50_2",
            ...     input_size=(256, 256),
            ...     layers=["layer1", "layer2", "layer3"],
            ...     anomaly_map_mode="multiply"
            ... )
            >>> input_tensor = torch.randn(1, 3, 256, 256)
            >>> # Training mode
            >>> model.train()
            >>> encoder_features, decoder_features = model(input_tensor)
            >>> # Evaluation mode
            >>> model.eval()
            >>> predictions = model(input_tensor)
        """
        self.encoder.eval()

        if self.tiler:
            images = self.tiler.tile(images)
        encoder_features = self.encoder(images)
        encoder_features = list(encoder_features.values())
        decoder_features = self.decoder(self.bottleneck(encoder_features))

        if self.tiler:
            for i, features in enumerate(encoder_features):
                encoder_features[i] = self.tiler.untile(features)
            for i, features in enumerate(decoder_features):
                decoder_features[i] = self.tiler.untile(features)

        if self.training:
            return encoder_features, decoder_features

        anomaly_map = self.anomaly_map_generator(encoder_features, decoder_features)
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
