"""Compute Anomaly map."""

# Original Code
# Copyright (c) 2022 hq-deng
# https://github.com/hq-deng/RD4AD
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Tuple, Union

import einops
import torch
import torch.nn.functional as F
from kornia.filters import gaussian_blur2d
from omegaconf import ListConfig
from torch import Tensor


class AnomalyMapGenerator:
    """Generate Anomaly Heatmap.

    Args:
        image_size (Union[ListConfig, Tuple]): Size of original image used for upscaling the anomaly map.
        sigma (int): Standard deviation of the gaussian kernel used to smooth anomaly map.
        mode (str, optional): Operation used to generate anomaly map. Options are `add` and `multiply`.
                Defaults to "multiply".

    Raises:
        ValueError: In case modes other than multiply and add are passed.
    """

    def __init__(self, image_size: Union[ListConfig, Tuple], sigma: int = 4, mode: str = "multiply"):
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)
        self.sigma = sigma
        self.kernel_size = 2 * int(4.0 * sigma + 0.5) + 1

        if mode not in ("add", "multiply"):
            raise ValueError(f"Found mode {mode}. Only multiply and add are supported.")
        self.mode = mode

        self.category_features = dict()

    def __call__(
        self, student_features: List[Tensor], teacher_features: List[Tensor]
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """Computes anomaly map given encoder and decoder features.

        Args:
            student_features (List[Tensor]): List of encoder features
            teacher_features (List[Tensor]): List of decoder features

        Returns:
            Tensor: Anomaly maps of length batch.
        """
        if self.mode == "multiply":
            anomaly_map = torch.ones(
                [student_features[0].shape[0], 1, *self.image_size], device=student_features[0].device
            )  # b c h w
        elif self.mode == "add":
            anomaly_map = torch.zeros(
                [student_features[0].shape[0], 1, *self.image_size], device=student_features[0].device
            )

        max_activation_val: Optional[Tensor] = None

        selected_featuremap = {k: [] for k in self.category_features.keys()}
        prev_len = 0

        for student_feature, teacher_feature in zip(student_features, teacher_features):
            distance_map = 1 - F.cosine_similarity(student_feature, teacher_feature)
            distance_map = torch.unsqueeze(distance_map, dim=1)
            distance_map = F.interpolate(distance_map, size=self.image_size, mode="bilinear", align_corners=True)

            # Use l1 norm to compute max activation values
            diff = torch.abs(F.normalize(student_feature) - F.normalize(teacher_feature))
            max_activations, _ = torch.max(einops.rearrange(diff, "b c h w -> b c (h w)"), -1)

            max_activation_val = (
                max_activations if max_activation_val is None else torch.hstack([max_activation_val, max_activations])
            )

            if self.mode == "multiply":
                anomaly_map *= distance_map
            elif self.mode == "add":
                anomaly_map += distance_map

            for name, _ in self.category_features.items():
                indices = (
                    self.category_features[name][
                        (self.category_features[name].long() >= prev_len)
                        & (self.category_features[name].long() < (prev_len + student_feature.shape[1]))
                    ].long()
                    - prev_len
                )
                selected_featuremap[name].append([student_feature[:, indices, ...], teacher_feature[:, indices, ...]])

        prev_len = student_feature.shape[1]
        anomaly_map = gaussian_blur2d(
            anomaly_map, kernel_size=(self.kernel_size, self.kernel_size), sigma=(self.sigma, self.sigma)
        )

        selected_featuremap = {
            k: list(zip(*[features for layer_tuple in layer_tuples for features in layer_tuple]))
            for k, layer_tuples in selected_featuremap.items()
        }

        return (anomaly_map, max_activation_val, selected_featuremap)

    def _compute_subclass_anomaly_map(self, distances: Tuple[Tensor]):
        distances = [(distances[0], distances[1]), (distances[2], distances[3]), (distances[4], distances[5])]
        if self.mode == "multiply":
            anomaly_map = torch.ones(
                [distances[0][0].shape[0], 1, *self.image_size], device=distances[0][0].device
            )  # b c h w
        elif self.mode == "add":
            anomaly_map = torch.zeros([distances[0][0].shape[0], 1, *self.image_size], device=distances[0][0].device)

        for student_feature, teacher_feature in distances:
            distance_map = 1 - F.cosine_similarity(student_feature, teacher_feature)
            distance_map = torch.unsqueeze(distance_map, dim=1)
            distance_map = F.interpolate(distance_map, size=self.image_size, mode="bilinear", align_corners=True)

            if self.mode == "multiply":
                anomaly_map *= distance_map
            elif self.mode == "add":
                anomaly_map += distance_map
        anomaly_map = gaussian_blur2d(
            anomaly_map, kernel_size=(self.kernel_size, self.kernel_size), sigma=(self.sigma, self.sigma)
        )
        return anomaly_map

    def feature_to_anomaly_map(self, distance: Tensor, feature: int) -> Tensor:
        if feature == -1:
            distance = tuple(dist.unsqueeze(0) for dist in distance)
        else:
            distance = tuple(dist[feature].reshape(1, 1, *dist[feature].shape) for dist in distance)
        return self._compute_subclass_anomaly_map(distance)
