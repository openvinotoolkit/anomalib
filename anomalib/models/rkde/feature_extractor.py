"""Region-based Anomaly Detection with Real Time Training and Analysis.

Region Extractor.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
from torchvision.ops import RoIAlign


class FeatureExtractor(nn.Module):
    """Region-based Feature Extractor."""

    def __init__(self) -> None:
        super().__init__()

        self.__model = models.alexnet(pretrained=False)

        # TODO: Load this via torch url.
        state_dict = torch.load("anomalib/models/rkde/rcnn_feature_extractor.pth", map_location="cpu")

        # Create the backbone.
        self.backbone = self.__model.features[:-1]
        self.backbone.load_state_dict(state_dict=state_dict["backbone"])

        # Create RoI Align Network.
        self.roi_align = RoIAlign(output_size=(6, 6), spatial_scale=1 / 16, sampling_ratio=0)

        # Classifier network to extract the features.
        self.classifer = self.__model.classifier[:-1]
        self.classifer.load_state_dict(state_dict=state_dict["classifier"])

    @torch.no_grad()
    def forward(self, input: Tensor, rois: Tensor) -> Tensor:
        """Forward-Pass Method.

        Args:
            input (Tensor): Input tensor of shape (N, C, H, W).
            rois (Tensor): Boxes tensor of shape (N, 4).

        Returns:
            Tensor: Region-based features extracted from the input tensor.
        """

        # Apply the feature extractor transforms
        input, scale = self.transform(input)

        # Scale the RoIs to the effective input size of the feature extractor.
        rois[:, 1:] *= scale

        # Forward-pass through the backbone, RoI Align and classifier.
        features = self.backbone(input)  # n_rois x 256 x 6 x 6 (AlexNet)
        features = self.roi_align(features, rois)  # n_rois x 4096
        features = self.classifer(features.view(features.size(0), -1))

        return features

    def transform(self, input: Tensor) -> Tuple[Tensor, float]:
        """Apply the feature extractor transforms

        Args:
            input (Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            Tuple[Tensor, float]: Output tensor of shape (N, C, H, W) and the scale.
        """
        height, width = input.shape[2:]
        shorter_image_size, longer_image_size = min(height, width), max(height, width)
        target_image_size, max_image_size = 600, 1000

        scale = target_image_size / shorter_image_size
        if round(scale * longer_image_size) > max_image_size:
            print("WARNING: cfg.MAX_SIZE exceeded. Using a different scaling ratio")
            scale = max_image_size / longer_image_size

        resized_tensor = F.interpolate(input, scale_factor=scale, mode="bilinear", align_corners=False)

        # Apply the same transformation as the original model.
        mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(input.device)
        std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(input.device)
        normalized_tensor = (resized_tensor - mean) / std

        return normalized_tensor, scale
