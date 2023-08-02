"""PyTorch model for Reverse Distillation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from torch import Tensor, nn

from anomalib.models.components import FeatureExtractor
from anomalib.models.reverse_distillation.anomaly_map import AnomalyMapGenerator
from anomalib.models.reverse_distillation.components import (
    get_bottleneck_layer,
    get_decoder,
)
from anomalib.pre_processing import Tiler


class ReverseDistillationModel(nn.Module):
    """Reverse Distillation Model.

    Args:
        backbone (str): Name of the backbone used for encoder and decoder
        input_size (tuple[int, int]): Size of input image
        layers (list[str]): Name of layers from which the features are extracted.
        anomaly_map_mode (str): Mode used to generate anomaly map. Options are between ``multiply`` and ``add``.
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone.
    """

    def __init__(
        self,
        backbone: str,
        input_size: tuple[int, int],
        layers: list[str],
        anomaly_map_mode: str,
        pre_trained: bool = True,
    ) -> None:
        super().__init__()
        self.tiler: Tiler | None = None

        encoder_backbone = backbone
        self.encoder = FeatureExtractor(backbone=encoder_backbone, pre_trained=pre_trained, layers=layers)
        self.bottleneck = get_bottleneck_layer(backbone)
        self.decoder = get_decoder(backbone)

        if self.tiler:
            image_size = (self.tiler.tile_size_h, self.tiler.tile_size_w)
        else:
            image_size = input_size

        self.anomaly_map_generator = AnomalyMapGenerator(image_size=image_size, mode=anomaly_map_mode)

    def forward(self, images: Tensor) -> Tensor | list[Tensor] | tuple[list[Tensor]]:
        """Forward-pass images to the network.

        During the training mode the model extracts features from encoder and decoder networks.
        During evaluation mode, it returns the predicted anomaly map.

        Args:
            images (Tensor): Batch of images

        Returns:
            Tensor | list[Tensor] | tuple[list[Tensor]]: Encoder and decoder features in training mode,
                else anomaly maps.
        """
        self.encoder.eval()

        if self.tiler:
            images = self.tiler.tile(images)
        encoder_features = self.encoder(images)
        encoder_features = list(encoder_features.values())
        decoder_features = self.decoder(self.bottleneck(encoder_features))

        if self.training:
            output = encoder_features, decoder_features
        else:
            output = self.anomaly_map_generator(encoder_features, decoder_features)
            if self.tiler:
                output = self.tiler.untile(output)

        return output
