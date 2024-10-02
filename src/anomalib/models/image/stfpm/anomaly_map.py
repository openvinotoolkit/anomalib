"""Anomaly Map Generator for the STFPM model implementation."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


class AnomalyMapGenerator(nn.Module):
    """Generate Anomaly Heatmap."""

    def __init__(self) -> None:
        super().__init__()
        self.distance = torch.nn.PairwiseDistance(p=2, keepdim=True)

    @staticmethod
    def compute_layer_map(
        teacher_features: torch.Tensor,
        student_features: torch.Tensor,
        image_size: tuple[int, int] | torch.Size,
    ) -> torch.Tensor:
        """Compute the layer map based on cosine similarity.

        Args:
          teacher_features (torch.Tensor): Teacher features
          student_features (torch.Tensor): Student features
          image_size (tuple[int, int]): Image size to which the anomaly map should be resized.

        Returns:
          Anomaly score based on cosine similarity.
        """
        norm_teacher_features = F.normalize(teacher_features)
        norm_student_features = F.normalize(student_features)

        layer_map = 0.5 * torch.norm(norm_teacher_features - norm_student_features, p=2, dim=-3, keepdim=True) ** 2
        return F.interpolate(layer_map, size=image_size, align_corners=False, mode="bilinear")

    def compute_anomaly_map(
        self,
        teacher_features: dict[str, torch.Tensor],
        student_features: dict[str, torch.Tensor],
        image_size: tuple[int, int] | torch.Size,
    ) -> torch.Tensor:
        """Compute the overall anomaly map via element-wise production the interpolated anomaly maps.

        Args:
          teacher_features (dict[str, torch.Tensor]): Teacher features
          student_features (dict[str, torch.Tensor]): Student features
          image_size (tuple[int, int]): Image size to which the anomaly map should be resized.

        Returns:
          Final anomaly map
        """
        batch_size = next(iter(teacher_features.values())).shape[0]
        anomaly_map = torch.ones(batch_size, 1, image_size[0], image_size[1])
        for layer in teacher_features:
            layer_map = self.compute_layer_map(teacher_features[layer], student_features[layer], image_size)
            anomaly_map = anomaly_map.to(layer_map.device)
            anomaly_map *= layer_map

        return anomaly_map

    def forward(self, **kwargs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return anomaly map.

        Expects `teach_features` and `student_features` keywords to be passed explicitly.

        Args:
            kwargs (dict[str, torch.Tensor]): Keyword arguments

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
            msg = f"Expected keys `teacher_features` and `student_features. Found {kwargs.keys()}"
            raise ValueError(msg)

        teacher_features: dict[str, torch.Tensor] = kwargs["teacher_features"]
        student_features: dict[str, torch.Tensor] = kwargs["student_features"]
        image_size: tuple[int, int] | torch.Size = kwargs["image_size"]

        return self.compute_anomaly_map(teacher_features, student_features, image_size)
