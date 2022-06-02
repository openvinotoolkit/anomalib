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

import torch
import torchvision
from torch import Tensor, nn

from anomalib.models.components import FeatureExtractor
from anomalib.models.reversedistillation.anomaly_map import AnomalyMapGenerator
from anomalib.models.reversedistillation.components import (
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
    """

    def __init__(self, backbone: str, input_size: Tuple[int, int], layers: List[str]):
        super().__init__()
        self.tiler: Optional[Tiler] = None

        encoder_backbone = getattr(torchvision.models, backbone)
        # TODO replace with TIMM feature extractor
        self.encoder = FeatureExtractor(backbone=encoder_backbone(pretrained=True), layers=layers)
        self.bottleneck = get_bottleneck_layer(backbone)
        self.encoder.eval()
        self.decoder = get_decoder(backbone)

        if self.tiler:
            image_size = (self.tiler.tile_size_h, self.tiler.tile_size_w)
        else:
            image_size = input_size

        self.anomaly_map_generator = AnomalyMapGenerator(image_size=tuple(image_size))

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
        if self.tiler:
            images = self.tiler.tile(images)
        encoder_features = self.encoder(images)
        encoder_features = list(encoder_features.values())
        decoder_features = self.decoder(self.bottleneck(encoder_features))

        if self.training:
            output = encoder_features, decoder_features
        else:
            output = self.anomaly_map_generator(encoder_features, decoder_features, amap_mode="add")
            if self.tiler:
                output = self.tiler.untile(output)

        return output

    def loss_function(self, encoder_features: List[Tensor], decoder_features: List[Tensor]) -> Tensor:
        """Computes cosine similarity loss based on features from encoder and decoder.

        Args:
            encoder_features (List[Tensor]): List of features extracted from encoder
            decoder_features (List[Tensor]): List of features extracted from decoder

        Returns:
            Tensor: Cosine similarity loss
        """
        cos_loss = torch.nn.CosineSimilarity()
        losses = list(map(cos_loss, encoder_features, decoder_features))
        loss_sum = 0
        for loss in losses:
            loss_sum += torch.mean(1 - loss)  # mean of cosine distance
        return loss_sum

    def get_loss(self, batch: Tensor) -> Tensor:
        """Computes cosine similarity loss given batch of images.

        Args:
            batch (Tensor): Batch of images

        Returns:
            Tensor: Cosine similarity loss
        """
        encoder_features, decoder_features = self.forward(batch)
        return self.loss_function(encoder_features, decoder_features)
