"""PyTorch model for Reverse Distillation."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import nn

from anomalib.models.components import TimmFeatureExtractor

from .anomaly_map import AnomalyMapGenerationMode, AnomalyMapGenerator
from .components import get_bottleneck_layer, get_decoder

if TYPE_CHECKING:
    from anomalib.data.utils.tiler import Tiler


class ReverseDistillationModel(nn.Module):
    """Reverse Distillation Model.

    To reproduce results in the paper, use torchvision model for the encoder:
        self.encoder = torchvision.models.wide_resnet50_2(pretrained=True)

    Args:
        backbone (str): Name of the backbone used for encoder and decoder.
        input_size (tuple[int, int]): Size of input image.
        layers (list[str]): Name of layers from which the features are extracted.
        anomaly_map_mode (str): Mode used to generate anomaly map. Options are between ``multiply`` and ``add``.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
            Defaults to ``True``.
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

    def forward(self, images: torch.Tensor) -> torch.Tensor | list[torch.Tensor] | tuple[list[torch.Tensor]]:
        """Forward-pass images to the network.

        During the training mode the model extracts features from encoder and decoder networks.
        During evaluation mode, it returns the predicted anomaly map.

        Args:
            images (torch.Tensor): Batch of images

        Returns:
            torch.Tensor | list[torch.Tensor] | tuple[list[torch.Tensor]]: Encoder and decoder features
                in training mode, else anomaly maps.
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
            output = encoder_features, decoder_features
        else:
            output = self.anomaly_map_generator(encoder_features, decoder_features)

        return output
