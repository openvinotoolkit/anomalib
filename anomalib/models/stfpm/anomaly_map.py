"""Anomaly Map Generator for the STFPM model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F
from omegaconf import ListConfig
from torch import Tensor, nn


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap."""

    def __init__(
        self,
        image_size: Union[ListConfig, Tuple],
    ):
        super().__init__()
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

    def forward(self, **kwargs: Dict[str, Tensor]) -> torch.Tensor:
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

        if not ("teacher_features" in kwargs and "student_features" in kwargs):
            raise ValueError(f"Expected keys `teacher_features` and `student_features. Found {kwargs.keys()}")

        teacher_features: Dict[str, Tensor] = kwargs["teacher_features"]
        student_features: Dict[str, Tensor] = kwargs["student_features"]

        return self.compute_anomaly_map(teacher_features, student_features)
