"""PyTorch model for Reverse Distillation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple, Union

from torch import Tensor, nn

from anomalib.models.components import get_feature_extractor
from anomalib.models.components.feature_extractors import FeatureExtractorParams
from anomalib.models.reverse_distillation.anomaly_map import AnomalyMapGenerator
from anomalib.models.reverse_distillation.components import (
    get_bottleneck_layer,
    get_decoder,
)
from anomalib.pre_processing import Tiler


class ReverseDistillationModel(nn.Module):
    """Reverse Distillation Model.

    Args:
        input_size (Tuple[int, int]): Size of input image
        anomaly_map_mode (str): Mode used to generate anomaly map. Options are between ``multiply`` and ``add``.
        feature_extractor (FeatureExtractorParams): Feature extractor params
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        anomaly_map_mode: str,
        feature_extractor: FeatureExtractorParams,
    ):
        super().__init__()
        self.tiler: Optional[Tiler] = None

        encoder_backbone = str(feature_extractor.backbone)
        self.encoder = get_feature_extractor(feature_extractor)
        self.bottleneck = get_bottleneck_layer(encoder_backbone)
        self.decoder = get_decoder(encoder_backbone)

        if self.tiler:
            image_size = (self.tiler.tile_size_h, self.tiler.tile_size_w)
        else:
            image_size = input_size

        self.anomaly_map_generator = AnomalyMapGenerator(image_size=tuple(image_size), mode=anomaly_map_mode)

    def forward(self, images: Tensor) -> Union[Tensor, Tuple[List[Tensor], List[Tensor]]]:
        """Forward-pass images to the network.

        During the training mode the model extracts features from encoder and decoder networks.
        During evaluation mode, it returns the predicted anomaly map.

        Args:
            images (Tensor): Batch of images

        Returns:
            Union[Tensor, Tuple[List[Tensor],List[Tensor]]]: Encoder and decoder features in training mode,
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
