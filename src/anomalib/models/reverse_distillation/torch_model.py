"""PyTorch model for Reverse Distillation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from torch import Tensor, nn

from anomalib.models.components import FeatureExtractor
from anomalib.models.reverse_distillation.anomaly_map import AnomalyMapGenerator
from anomalib.models.reverse_distillation.components import get_bottleneck_layer, get_decoder
from anomalib.pre_processing import Tiler

from .anomaly_map import AnomalyMapGenerationMode


class ReverseDistillationModel(nn.Module):
    """Reverse Distillation Model.

    Args:
        input_size (tuple[int, int], optional): Size of model input. Defaults to (256, 256).
        backbone (str, optional): Backbone of CNN network. Defaults to "wide_resnet50_2".
        layers (list[str], optional): Layers to extract features from the backbone CNN.
            Defaults to ["layer1", "layer2", "layer3"].
        pre_trained (bool, optional): Boolean to check whether to use a pre_trained backbone. Defaults to True.
    """

    def __init__(
        self,
        input_size: tuple[int, int] = (256, 256),
        backbone: str = "wide_resnet50_2",
        layers: list[str] = ["layer1", "layer2", "layer3"],
        anomaly_map_mode: AnomalyMapGenerationMode = AnomalyMapGenerationMode.MULTIPLY,
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
