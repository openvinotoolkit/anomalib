"""Region-based Anomaly Detection with Real Time Training and Analysis.

Feature Extractor.
"""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torchvision.ops import RoIAlign
from torchvision.transforms import Normalize, Resize

from anomalib.data.utils.boxes import scale_boxes

WEIGHTS_URL = "https://github.com/openvinotoolkit/anomalib/releases/download/rkde-weights/rkde_feature_extractor.pth"


class FeatureExtractor(nn.Module):
    """Feature Extractor module for Region-based anomaly detection."""

    def __init__(self) -> None:
        super().__init__()

        self.transform = nn.Sequential(
            Resize(size=600, max_size=1000),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.roi_align = RoIAlign(output_size=(6, 6), spatial_scale=1 / 16, sampling_ratio=0)

        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

        # load the pre-trained weights from url
        self.load_state_dict(torch.hub.load_state_dict_from_url(WEIGHTS_URL, progress=False))

    @torch.no_grad()
    def forward(self, batch: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the feature extractor.

        Args:
            batch (torch.Tensor): Batch of input images of shape [B, C, H, W].
            rois (torch.Tensor): torch.Tensor of shape [N, 5] describing the regions-of-interest in the batch.

        Returns:
            Tensor: torch.Tensor containing a 4096-dimensional feature vector for every RoI location.
        """
        # Apply the feature extractor transforms
        transformed_batch = self.transform(batch)

        # Scale the RoIs to the effective input size of the feature extractor.
        rois[:, 1:] = scale_boxes(rois[:, 1:], batch.shape[-2:], transformed_batch.shape[-2:])

        # Forward pass through the backbone
        features = self.features(transformed_batch)
        features = self.roi_align(features, rois)
        features = torch.flatten(features, 1)
        return self.classifier(features)
