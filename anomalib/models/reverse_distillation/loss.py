"""Loss function for Reverse Distillation."""

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

from typing import List

import torch
from torch import Tensor


class ReverseDistillationLoss:
    """Loss function for Reverse Distillation."""

    def __call__(self, encoder_features: List[Tensor], decoder_features: List[Tensor]) -> Tensor:
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
