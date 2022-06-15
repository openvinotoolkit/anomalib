"""PyTorch model for Reverse Distillation."""

# Copyright (C) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import List, Optional, Tuple, Union

import torchvision
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
        input_size (Tuple[int, int]): Size of input image
        layers (List[str]): Name of layers from which the features are extracted.
        anomaly_map_mode (str): Mode used to generate anomaly map. Options are between ``multiply`` and ``add``.
    """

    def __init__(self, backbone: str, input_size: Tuple[int, int], layers: List[str], anomaly_map_mode: str):
        super().__init__()
        self.tiler: Optional[Tiler] = None

        encoder_backbone = getattr(torchvision.models, backbone)
        # TODO replace with TIMM feature extractor
        self.encoder = FeatureExtractor(backbone=encoder_backbone(pretrained=True), layers=layers)
        self.bottleneck = get_bottleneck_layer(backbone)
        self.decoder = get_decoder(backbone)

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
