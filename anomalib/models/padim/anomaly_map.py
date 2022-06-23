"""Anomaly Map Generator for the PaDiM model implementation."""

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

from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from omegaconf import ListConfig
from torch import Tensor


class AnomalyMapGenerator:
    """Generate Anomaly Heatmap.

    Args:
        image_size (Union[ListConfig, Tuple]): Size of the input image. The anomaly map is upsampled to this dimension.
        sigma (int, optional): Standard deviation for Gaussian Kernel. Defaults to 4.
    """

    def __init__(self, image_size: Union[ListConfig, Tuple], sigma: int = 4):
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)
        self.sigma = sigma

        # save indices of top features used for sub classification.
        # these features are selected for visualization
        self.category_features = dict()

    @staticmethod
    def compute_distance(
        embedding: Tensor, stats: List[Tensor], category_features: Dict[str, Tensor]
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """Compute anomaly score to the patch in position(i,j) of a test image.

        Ref: Equation (2), Section III-C of the paper.

        Args:
            embedding (Tensor): Embedding Vector
            stats (List[Tensor]): Mean and Covariance Matrix of the multivariate Gaussian distribution

        Returns:
            Anomaly score of a test image via mahalanobis distance.
        """

        batch, channel, height, width = embedding.shape
        embedding = embedding.reshape(batch, channel, height * width)

        # calculate mahalanobis distances
        mean, inv_covariance = stats

        delta = (embedding - mean).permute(2, 0, 1)

        distances_full = (torch.matmul(delta, inv_covariance) * delta).sum(2).permute(1, 0)
        distances_full = distances_full.reshape(batch, height, width)
        distances_full = torch.sqrt(distances_full)
        max_activation_val, _ = torch.max((torch.matmul(delta, inv_covariance) * delta), 0)

        distances = (torch.matmul(delta, inv_covariance) * delta).permute(1, 2, 0)
        distances = distances.reshape(batch, channel, height, width)

        selected_features = {}
        if category_features != {}:
            for name, keep_idx in category_features.items():
                selected_features[name] = distances[:, keep_idx.long(), :]

        return (distances_full, max_activation_val, selected_features)

    def up_sample(self, distance: Tensor) -> Tensor:
        """Up sample anomaly score to match the input image size.

        Args:
            distance (Tensor): Anomaly score computed via the mahalanobis distance.

        Returns:
            Resized distance matrix matching the input image size
        """

        score_map = F.interpolate(
            distance.unsqueeze(1),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )
        return score_map

    def smooth_anomaly_map(self, anomaly_map: Tensor) -> Tensor:
        """Apply gaussian smoothing to the anomaly map.

        Args:
            anomaly_map (Tensor): Anomaly score for the test image(s).

        Returns:
            Filtered anomaly scores
        """

        kernel_size = 2 * int(4.0 * self.sigma + 0.5) + 1
        sigma = torch.as_tensor(self.sigma).to(anomaly_map.device)
        anomaly_map = gaussian_blur2d(anomaly_map, (kernel_size, kernel_size), sigma=(sigma, sigma))

        return anomaly_map

    def compute_and_smooth(
        self, embedding: Tensor, mean: Tensor, inv_covariance: Tensor
    ) -> Tuple[Tensor, Tensor, Dict]:
        """Compute anomaly score.

        Scores are calculated based on embedding vector, mean and inv_covariance of the multivariate gaussian
        distribution.

        Args:
            embedding (Tensor): Embedding vector extracted from the test set.
            mean (Tensor): Mean of the multivariate gaussian distribution
            inv_covariance (Tensor): Inverse Covariance matrix of the multivariate gaussian distribution.

        Returns:
            Tuple: Anomaly map and max_activation value
        """
        score_map, max_activation_val, selected_features = self.compute_distance(
            embedding=embedding, stats=[mean, inv_covariance], category_features=self.category_features
        )
        up_sampled_score_map = self.up_sample(score_map)
        smoothed_anomaly_map = self.smooth_anomaly_map(up_sampled_score_map)

        return (smoothed_anomaly_map, max_activation_val, selected_features)

    def compute_anomaly_map(
        self, embedding: Tensor, mean: Tensor, inv_covariance: Tensor
    ) -> Tuple[Tensor, Tensor, Dict]:
        """Compute anomaly score.

        Scores are calculated based on embedding vector, mean and inv_covariance of the multivariate gaussian
        distribution.

        Args:
            embedding (Tensor): Embedding vector extracted from the test set.
            mean (Tensor): Mean of the multivariate gaussian distribution
            inv_covariance (Tensor): Inverse Covariance matrix of the multivariate gaussian distribution.

        Returns:
            Tuple: Anomaly map, Max anomaly score, Anomaly map from selected features
        """

        mean = mean.to(embedding.device)
        inv_covariance = inv_covariance.to(embedding.device)

        smoothed_anomaly_map, max_activation_val, selected_features = self.compute_and_smooth(
            embedding, mean, inv_covariance
        )
        return (smoothed_anomaly_map, max_activation_val, selected_features)

    def __call__(self, **kwds):
        """Returns anomaly_map.

        Expects `embedding`, `mean` and `covariance` keywords to be passed explicitly.

        Example:
        >>> anomaly_map_generator = AnomalyMapGenerator(image_size=input_size)
        >>> output = anomaly_map_generator(embedding=embedding, mean=mean, covariance=covariance)

        Raises:
            ValueError: `embedding`. `mean` or `covariance` keys are not found

        Returns:
            torch.Tensor: anomaly map
        """

        if not ("embedding" in kwds and "mean" in kwds and "inv_covariance" in kwds):
            raise ValueError(f"Expected keys `embedding`, `mean` and `covariance`. Found {kwds.keys()}")

        embedding: Tensor = kwds["embedding"]
        mean: Tensor = kwds["mean"]
        inv_covariance: Tensor = kwds["inv_covariance"]

        return self.compute_anomaly_map(embedding, mean, inv_covariance)
