"""PyTorch model implementation for Student-Teacher Feature Pyramid Matching.

This module implements the core PyTorch model architecture for the STFPM anomaly
detection method as described in `Wang et al. (2021)
<https://arxiv.org/abs/2103.04257>`_.

The model consists of:
- A pre-trained teacher network that extracts multi-scale features
- A student network that learns to match the teacher's feature representations
- Feature pyramid matching between student and teacher features
- Anomaly detection based on feature discrepancy

Example:
    >>> from anomalib.models.image.stfpm.torch_model import STFPMModel
    >>> model = STFPMModel(
    ...     backbone="resnet18",
    ...     layers=["layer1", "layer2", "layer3"]
    ... )
    >>> features = model(torch.randn(1, 3, 256, 256))

See Also:
    - :class:`STFPMModel`: Main PyTorch model implementation
    - :class:`STFPMLoss`: Loss function for training
    - :class:`AnomalyMapGenerator`: Anomaly map generation from features
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import nn

from anomalib.data import InferenceBatch
from anomalib.models.components import TimmFeatureExtractor

from .anomaly_map import AnomalyMapGenerator

if TYPE_CHECKING:
    from anomalib.data.utils.tiler import Tiler


class STFPMModel(nn.Module):
    """PyTorch implementation of the STFPM model.

    The Student-Teacher Feature Pyramid Matching model consists of a pre-trained
    teacher network and a student network that learns to match the teacher's
    feature representations. The model detects anomalies by comparing feature
    discrepancies between the teacher and student networks.

    Args:
        layers (Sequence[str]): Names of layers from which to extract features.
            For example ``["layer1", "layer2", "layer3"]``.
        backbone (str, optional): Name of the backbone CNN architecture used for
            both teacher and student networks. Supported backbones can be found
            in timm library. Defaults to ``"resnet18"``.

    Example:
        >>> import torch
        >>> from anomalib.models.image.stfpm.torch_model import STFPMModel
        >>> model = STFPMModel(
        ...     backbone="resnet18",
        ...     layers=["layer1", "layer2", "layer3"]
        ... )
        >>> input_tensor = torch.randn(1, 3, 256, 256)
        >>> features = model(input_tensor)

    Note:
        The teacher model is initialized with pre-trained weights and frozen
        during training, while the student model is trained from scratch.

    Attributes:
        tiler (Tiler | None): Optional tiler for processing large images in
            patches.
        teacher_model (TimmFeatureExtractor): Pre-trained teacher network for
            feature extraction.
        student_model (TimmFeatureExtractor): Student network that learns to
            match teacher features.
        anomaly_map_generator (AnomalyMapGenerator): Module to generate anomaly
            maps from features.
    """

    def __init__(
        self,
        layers: Sequence[str],
        backbone: str = "resnet18",
    ) -> None:
        super().__init__()
        self.tiler: Tiler | None = None

        self.backbone = backbone
        self.teacher_model = TimmFeatureExtractor(backbone=self.backbone, pre_trained=True, layers=layers).eval()
        self.student_model = TimmFeatureExtractor(
            backbone=self.backbone,
            pre_trained=False,
            layers=layers,
            requires_grad=True,
        )

        # teacher model is fixed
        for parameters in self.teacher_model.parameters():
            parameters.requires_grad = False

        self.anomaly_map_generator = AnomalyMapGenerator()

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]] | InferenceBatch:
        """Forward pass through teacher and student networks.

        The forward pass behavior differs between training and evaluation:
        - Training: Returns features from both teacher and student networks
        - Evaluation: Returns anomaly maps generated from feature differences

        Args:
            images (torch.Tensor): Batch of input images with shape
                ``(N, C, H, W)``.

        Returns:
            Training mode:
                tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
                    Features from teacher and student networks respectively.
                    Each dict maps layer names to feature tensors.
            Evaluation mode:
                InferenceBatch:
                    Batch containing anomaly maps and prediction scores.

        Example:
            >>> import torch
            >>> from anomalib.models.image.stfpm.torch_model import STFPMModel
            >>> model = STFPMModel(layers=["layer1", "layer2", "layer3"])
            >>> input_tensor = torch.randn(1, 3, 256, 256)
            >>> # Training mode
            >>> model.train()
            >>> teacher_feats, student_feats = model(input_tensor)
            >>> # Evaluation mode
            >>> model.eval()
            >>> predictions = model(input_tensor)
        """
        output_size = images.shape[-2:]
        if self.tiler:
            images = self.tiler.tile(images)
        teacher_features: dict[str, torch.Tensor] = self.teacher_model(images)
        student_features: dict[str, torch.Tensor] = self.student_model(images)

        if self.tiler:
            for layer, data in teacher_features.items():
                teacher_features[layer] = self.tiler.untile(data)
            for layer, data in student_features.items():
                student_features[layer] = self.tiler.untile(data)

        if self.training:
            return teacher_features, student_features

        anomaly_map = self.anomaly_map_generator(
            teacher_features=teacher_features,
            student_features=student_features,
            image_size=output_size,
        )
        pred_score = torch.amax(anomaly_map, dim=(-2, -1))
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
