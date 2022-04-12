"""Anomaly Map Generator for the STFPM model implementation."""

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

from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from torch import Tensor


class AnomalyMapGenerator:
    """Generate Anomaly Heatmap."""

    def __init__(
        self,
        image_size: Union[ListConfig, Tuple],
    ):
        self.distance = torch.nn.PairwiseDistance(p=2, keepdim=True)
        self.image_size = image_size if isinstance(image_size, tuple) else tuple(image_size)

    def compute_layer_map(self, teacher_features: Tensor, student_features: Tensor) -> Tensor:
        """Compute the layer map based on cosine similarity.

        Args:
          teacher_features (Tensor): Teacher features
          student_features (Tensor): Student features

        Returns:
          Anomaly score based on cosine similarity.
        """
        norm_teacher_features = F.normalize(teacher_features)
        norm_student_features = F.normalize(student_features)

        layer_map = 0.5 * torch.norm(norm_teacher_features - norm_student_features, p=2, dim=-3, keepdim=True) ** 2
        layer_map = F.interpolate(layer_map, size=self.image_size, align_corners=False, mode="bilinear")
        return layer_map

    def compute_anomaly_map(
        self, teacher_features: Dict[str, Tensor], student_features: Dict[str, Tensor]
    ) -> torch.Tensor:
        """Compute the overall anomaly map via element-wise production the interpolated anomaly maps.

        Args:
          teacher_features (Dict[str, Tensor]): Teacher features
          student_features (Dict[str, Tensor]): Student features

        Returns:
          Final anomaly map
        """
        batch_size = list(teacher_features.values())[0].shape[0]
        anomaly_map = torch.ones(batch_size, 1, self.image_size[0], self.image_size[1])
        for layer in teacher_features.keys():
            layer_map = self.compute_layer_map(teacher_features[layer], student_features[layer])
            anomaly_map = anomaly_map.to(layer_map.device)
            anomaly_map *= layer_map

        return anomaly_map

    def __call__(self, **kwds: Dict[str, Tensor]) -> torch.Tensor:
        """Returns anomaly map.

        Expects `teach_features` and `student_features` keywords to be passed explicitly.

        Example:
            >>> anomaly_map_generator = AnomalyMapGenerator(image_size=tuple(hparams.model.input_size))
            >>> output = self.anomaly_map_generator(
                    teacher_features=teacher_features,
                    student_features=student_features
                )

        Raises:
            ValueError: `teach_features` and `student_features` keys are not found

        Returns:
            torch.Tensor: anomaly map
        """

        if not ("teacher_features" in kwds and "student_features" in kwds):
            raise ValueError(f"Expected keys `teacher_features` and `student_features. Found {kwds.keys()}")

        teacher_features: Dict[str, Tensor] = kwds["teacher_features"]
        student_features: Dict[str, Tensor] = kwds["student_features"]

        return self.compute_anomaly_map(teacher_features, student_features)
