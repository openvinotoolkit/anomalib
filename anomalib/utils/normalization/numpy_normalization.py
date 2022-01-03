"""Normalization functions Numpy for post-processing."""

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

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import norm


class CDFNormalization:
    """Encapsulates static methods for computing CDF Normalization."""

    @staticmethod
    def standardize(
        pred_scores: np.ndarray, stats: Dict, anomaly_map: Optional[np.ndarray] = None
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Standardize the predicted scores and anomaly maps to the z-domain.

        Args:
            pred_scores (np.ndarray): Scores predicted by the model.
            stats (AnomalyScoreDistribution): training distribution which stores image/pixel mean and std.
            anomaly_map (Optional[np.ndarray], optional): Anomaly map. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: anomaly maps and predicted scores standardized to z-domain.
        """
        if "pixel_mean" in stats.keys() and "pixel_std" in stats.keys():
            anomaly_map = np.log(anomaly_map)
            anomaly_map = (anomaly_map - stats["pixel_mean"]) / stats["pixel_std"]
            anomaly_map -= (stats["image_mean"] - stats["pixel_mean"]) / stats["pixel_std"]

        # standardize image scores
        if "image_mean" in stats.keys() and "image_std" in stats.keys():
            pred_scores = np.log(pred_scores)
            pred_scores = (pred_scores - stats["image_mean"]) / stats["image_std"]

        return anomaly_map, pred_scores

    @staticmethod
    def normalize(
        pred_scores: np.ndarray,
        image_threshold: np.ndarray,
        pixel_threshold: np.ndarray = None,
        anomaly_map: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Normalize the predicted scores and anomaly maps.

        Args:
            pred_scores (Tensor): Standardized scores
            image_threshold (Union[Module, Tensor]): Threshold for normalizing the scores
            pixel_threshold (Optional[Union[Module, Tensor]], optional): Threshold for normalizing the anomaly maps.
                                                                        Defaults to None.
            anomaly_maps (Optional[Tensor], optional): Standardized anomaly map. Defaults to None.

        Returns:
            Tuple[Optional[Tensor], Tensor]: Normalized anomaly maps and predicted scores
        """
        pred_scores = norm.cdf(pred_scores - image_threshold)
        if anomaly_map is not None:
            anomaly_map = norm.cdf(anomaly_map - pixel_threshold)

        return anomaly_map, pred_scores
