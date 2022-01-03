"""Normalization functions PyTorch for post-processing."""

# Copyright (C) 2020 Intel Corporation
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

from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.distributions import Normal
from torch.nn import Module

from anomalib.core.metrics import AnomalyScoreDistribution


class CDFNormalization:
    """Encapsulates static methods for computing CDF Normalization."""

    @staticmethod
    def standardize(
        pred_scores: Tensor, stats: AnomalyScoreDistribution, anomaly_maps: Optional[Tensor] = None
    ) -> Tuple[Optional[Tensor], Tensor]:
        """Standardize the predicted scores and anomaly maps to the z-domain.

        Args:
            pred_scores (Tensor): Scores predicted by the model.
            stats (AnomalyScoreDistribution): training distribution which stores image/pixel mean and std.
            anomaly_maps (Optional[Tensor], optional): Anomaly maps. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: anomaly maps and predicted scores standardized to z-domain.
        """
        pred_scores = torch.log(pred_scores)
        pred_scores = (pred_scores - stats.image_mean) / stats.image_std
        if anomaly_maps is not None:
            anomaly_maps = (torch.log(anomaly_maps) - stats.pixel_mean) / stats.pixel_std
            anomaly_maps -= (stats.image_mean - stats.pixel_mean) / stats.pixel_std

        return anomaly_maps, pred_scores

    @staticmethod
    def normalize(
        pred_scores: Tensor,
        image_threshold: Union[Module, Tensor],
        pixel_threshold: Optional[Union[Module, Tensor]] = None,
        anomaly_maps: Optional[Tensor] = None,
    ) -> Tuple[Optional[Tensor], Tensor]:
        """Normalize the predicted scores and anomaly maps.

        Args:
            pred_scores (Tensor): Standardized scores
            image_threshold (Union[Module, Tensor]): Threshold for normalizing the scores
            pixel_threshold (Optional[Union[Module, Tensor]], optional): Threshold for normalizing the anomaly maps.
                                                                        Defaults to None.
            anomaly_maps (Optional[Tensor], optional): Standardized anomaly maps. Defaults to None.

        Returns:
            Tuple[Optional[Tensor], Tensor]: Normalized anomaly maps and predicted scores
        """
        device = pred_scores.device
        norm = Normal(Tensor([0]), Tensor([1]))
        pred_scores = norm.cdf(pred_scores.cpu() - image_threshold).to(device)
        if anomaly_maps is not None:
            anomaly_maps = norm.cdf(anomaly_maps.cpu() - pixel_threshold).to(device)

        return anomaly_maps, pred_scores
