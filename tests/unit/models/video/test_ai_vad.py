"""Unit tests for AI-VAD video anomaly detection model."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import torch

from anomalib.models.video.ai_vad import AiVad


class TestAiVadFeatureExtractor:
    """Test if the different feature extractors of the AiVad model can handle edge case without bbox detections."""

    @staticmethod
    def test_velocity_extractor() -> None:
        """Test velocity extractor submodule."""
        pl_module = AiVad()
        velocity_feature_extractor = pl_module.model.feature_extractor.velocity_extractor

        flow_data = torch.zeros(4, 2, 256, 256)
        boxes = torch.empty(0, 5)

        velocity_features = velocity_feature_extractor(flow_data, boxes)

        # features should be empty because there are no boxes
        assert velocity_features.numel() == 0

    @staticmethod
    def test_deep_feature_extractor() -> None:
        """Test deep feature extractor submodule."""
        pl_module = AiVad()
        deep_feature_extractor = pl_module.model.feature_extractor.deep_extractor

        batch_size = 4
        rgb_data = torch.zeros(batch_size, 3, 256, 256)
        boxes = torch.empty(0, 5)

        deep_features = deep_feature_extractor(rgb_data, boxes, batch_size)

        # features should be empty because there are no boxes
        assert deep_features.numel() == 0

    @staticmethod
    def test_pose_feature_extractor() -> None:
        """Test pose feature extractor submodule."""
        pl_module = AiVad()
        pose_feature_extractor = pl_module.model.feature_extractor.pose_extractor

        batch_size = 4
        rgb_data = torch.zeros(batch_size, 3, 256, 256)
        boxes = [torch.empty(0, 4)] * batch_size

        pose_features = pose_feature_extractor(rgb_data, boxes)

        # features should be empty because there are no boxes
        assert torch.vstack(pose_features).numel() == 0
