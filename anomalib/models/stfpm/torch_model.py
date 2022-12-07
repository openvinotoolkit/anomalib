"""PyTorch model for the STFPM model implementation."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, Tuple

from torch import Tensor, nn

from anomalib.models.components import get_feature_extractor
from anomalib.models.components.feature_extractors import (
    FeatureExtractorParams,
    TimmFeatureExtractorParams,
    TorchFXFeatureExtractorParams,
)
from anomalib.models.components.feature_extractors.utils import _convert_datatype
from anomalib.models.stfpm.anomaly_map import AnomalyMapGenerator
from anomalib.pre_processing import Tiler


class STFPMModel(nn.Module):
    """STFPM: Student-Teacher Feature Pyramid Matching for Unsupervised Anomaly Detection.

    Args:
        layers (List[str]): Layers used for feature extraction
        input_size (Tuple[int, int]): Input size for the model.
        student_teacher_model (FeatureExtractorParams): Teacher model parameters.
    """

    def __init__(self, input_size: Tuple[int, int], student_teacher_model: FeatureExtractorParams):
        super().__init__()
        self.tiler: Optional[Tiler] = None

        self.teacher_model: nn.Module
        self.student_model: nn.Module
        self._initialize_models(student_teacher_model)

        # teacher model is fixed
        for parameters in self.teacher_model.parameters():
            parameters.requires_grad = False

        # Create the anomaly heatmap generator whether tiling is set.
        # TODO: Check whether Tiler is properly initialized here.
        if self.tiler:
            image_size = (self.tiler.tile_size_h, self.tiler.tile_size_w)
        else:
            image_size = input_size
        self.anomaly_map_generator = AnomalyMapGenerator(image_size=tuple(image_size))

    def _initialize_models(self, student_teacher_model: FeatureExtractorParams):
        """Initialize the teacher and student models.

        Args:
            student_teacher_model (FeatureExtractorParams): Model parameters.
        """
        # When loading from the entrypoint scripts student_teacher_model is DictConfig
        student_teacher_model = _convert_datatype(student_teacher_model)
        teacher_model_params: FeatureExtractorParams
        student_model_params: FeatureExtractorParams
        if isinstance(student_teacher_model, TimmFeatureExtractorParams):
            teacher_model_params = TimmFeatureExtractorParams(
                backbone=student_teacher_model.backbone,
                layers=student_teacher_model.layers,
                pre_trained=True,
                requires_grad=False,
            )
            student_model_params = TimmFeatureExtractorParams(
                backbone=student_teacher_model.backbone,
                layers=student_teacher_model.layers,
                pre_trained=False,
                requires_grad=True,
            )
        else:
            teacher_model_params = TorchFXFeatureExtractorParams(
                backbone=student_teacher_model.backbone,
                return_nodes=student_teacher_model.return_nodes,
                weights=student_teacher_model.weights,
                requires_grad=False,
            )
            student_model_params = TorchFXFeatureExtractorParams(
                backbone=student_teacher_model.backbone,
                return_nodes=student_teacher_model.return_nodes,
                weights=None,
                requires_grad=True,
            )
        self.teacher_model = get_feature_extractor(teacher_model_params)
        self.student_model = get_feature_extractor(student_model_params)

    def forward(self, images):
        """Forward-pass images into the network.

        During the training mode the model extracts the features from the teacher and student networks.
        During the evaluation mode, it returns the predicted anomaly map.

        Args:
          images (Tensor): Batch of images.

        Returns:
          Teacher and student features when in training mode, otherwise the predicted anomaly maps.
        """
        if self.tiler:
            images = self.tiler.tile(images)
        teacher_features: Dict[str, Tensor] = self.teacher_model(images)
        student_features: Dict[str, Tensor] = self.student_model(images)
        if self.training:
            output = teacher_features, student_features
        else:
            output = self.anomaly_map_generator(teacher_features=teacher_features, student_features=student_features)
            if self.tiler:
                output = self.tiler.untile(output)

        return output
