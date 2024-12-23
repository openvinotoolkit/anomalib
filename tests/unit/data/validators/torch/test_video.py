"""Test Torch Video Validators."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from torchvision.tv_tensors import Mask

from anomalib.data.validators.torch.video import VideoBatchValidator, VideoValidator


class TestVideoValidator:
    """Test VideoValidator."""

    def setup_method(self) -> None:
        """Set up the validator for each test method."""
        self.validator = VideoValidator()

    def test_validate_image_valid(self) -> None:
        """Test validation of a valid video tensor."""
        video = torch.rand(10, 3, 224, 224)
        validated_video = self.validator.validate_image(video)
        assert isinstance(validated_video, torch.Tensor)
        assert validated_video.shape == (10, 3, 224, 224)
        assert validated_video.dtype == torch.float32

    def test_validate_image_invalid_type(self) -> None:
        """Test validation of a video tensor with invalid type."""
        with pytest.raises(TypeError, match="Video must be a torch.Tensor"):
            self.validator.validate_image(np.random.default_rng().random((10, 3, 224, 224)))

    def test_validate_image_invalid_dimensions(self) -> None:
        """Test validation of a video tensor with invalid dimensions."""
        with pytest.raises(ValueError, match="Video must have 3 or 4 dimensions"):
            self.validator.validate_image(torch.rand(224, 224))

    def test_validate_image_invalid_channels(self) -> None:
        """Test validation of a video tensor with invalid number of channels."""
        with pytest.raises(ValueError, match="Video must have 1 or 3 channels"):
            self.validator.validate_image(torch.rand(10, 2, 224, 224))

    def test_validate_gt_label_valid(self) -> None:
        """Test validation of a valid ground truth label."""
        label = torch.tensor(1)
        validated_label = self.validator.validate_gt_label(label)
        assert isinstance(validated_label, torch.Tensor)
        assert validated_label.dtype == torch.bool
        assert validated_label.item() is True

    def test_validate_gt_label_none(self) -> None:
        """Test validation of a None ground truth label."""
        assert self.validator.validate_gt_label(None) is None

    def test_validate_gt_label_invalid_type(self) -> None:
        """Test validation of a ground truth label with invalid type."""
        with pytest.raises(TypeError, match="Ground truth label must be an integer or a torch.Tensor"):
            self.validator.validate_gt_label("1")

    def test_validate_gt_mask_valid(self) -> None:
        """Test validation of a valid ground truth mask."""
        mask = torch.randint(0, 2, (10, 1, 224, 224))
        validated_mask = self.validator.validate_gt_mask(mask)
        assert isinstance(validated_mask, Mask)
        assert validated_mask.shape == (10, 224, 224)
        assert validated_mask.dtype == torch.bool

    def test_validate_gt_mask_none(self) -> None:
        """Test validation of a None ground truth mask."""
        assert self.validator.validate_gt_mask(None) is None

    def test_validate_gt_mask_invalid_type(self) -> None:
        """Test validation of a ground truth mask with invalid type."""
        with pytest.raises(TypeError, match="Mask must be a torch.Tensor"):
            self.validator.validate_gt_mask(np.random.default_rng().integers(0, 2, (10, 224, 224)))

    def test_validate_gt_mask_invalid_shape(self) -> None:
        """Test validation of a ground truth mask with invalid shape."""
        with pytest.raises(ValueError, match="Mask must have 1 channel, got 2."):
            self.validator.validate_gt_mask(torch.randint(0, 2, (10, 2, 224, 224)))

    def test_validate_anomaly_map_valid(self) -> None:
        """Test validation of a valid anomaly map."""
        anomaly_map = torch.rand(10, 1, 224, 224)
        validated_map = self.validator.validate_anomaly_map(anomaly_map)
        assert isinstance(validated_map, Mask)
        assert validated_map.shape == (10, 224, 224)
        assert validated_map.dtype == torch.float32

    def test_validate_anomaly_map_none(self) -> None:
        """Test validation of a None anomaly map."""
        assert self.validator.validate_anomaly_map(None) is None

    def test_validate_anomaly_map_invalid_type(self) -> None:
        """Test validation of an anomaly map with invalid type."""
        with pytest.raises(TypeError, match="Anomaly map must be a torch.Tensor"):
            self.validator.validate_anomaly_map(np.random.default_rng().random((10, 224, 224)))

    def test_validate_anomaly_map_invalid_shape(self) -> None:
        """Test validation of an anomaly map with invalid shape."""
        with pytest.raises(ValueError, match="Anomaly map with 4 dimensions must have 1 channel, got 2."):
            self.validator.validate_anomaly_map(torch.rand(10, 2, 224, 224))

    def test_validate_pred_score_valid(self) -> None:
        """Test validation of a valid prediction score."""
        score = torch.tensor(0.8)
        validated_score = self.validator.validate_pred_score(score)
        assert isinstance(validated_score, torch.Tensor)
        assert validated_score.dtype == torch.float32
        assert validated_score.item() == pytest.approx(0.8)

    def test_validate_pred_score_none(self) -> None:
        """Test validation of a None prediction score."""
        assert self.validator.validate_pred_score(None) is None

    def test_validate_pred_score_invalid_shape(self) -> None:
        """Test validation of a prediction score with invalid shape."""
        with pytest.raises(ValueError, match="Predicted score must be a scalar"):
            self.validator.validate_pred_score(torch.tensor([0.8, 0.9]))


class TestVideoBatchValidator:
    """Test VideoBatchValidator."""

    def setup_method(self) -> None:
        """Set up the validator for each test method."""
        self.validator = VideoBatchValidator()

    def test_validate_image_valid(self) -> None:
        """Test validation of a valid video batch."""
        video_batch = torch.rand(2, 10, 3, 224, 224)
        validated_batch = self.validator.validate_image(video_batch)
        assert validated_batch.shape == (2, 10, 3, 224, 224)
        assert validated_batch.dtype == torch.float32

    def test_validate_image_invalid_type(self) -> None:
        """Test validation of a video batch with invalid type."""
        with pytest.raises(TypeError, match="Video batch must be a torch.Tensor"):
            self.validator.validate_image(np.random.default_rng().random((2, 10, 3, 224, 224)))

    def test_validate_image_invalid_dimensions(self) -> None:
        """Test validation of a video batch with invalid dimensions."""
        with pytest.raises(
            ValueError,
            match=(
                r"Video batch must have 4 dimensions \(B, C, H, W\) for single frame images or "
                r"5 dimensions \(B, T, C, H, W\) for multi-frame videos, got 2."
            ),
        ):
            self.validator.validate_image(torch.rand(224, 224))

    def test_validate_image_invalid_channels(self) -> None:
        """Test validation of a video batch with invalid number of channels."""
        with pytest.raises(ValueError, match="Video batch must have 1 or 3 channels"):
            self.validator.validate_image(torch.rand(2, 10, 2, 224, 224))

    def test_validate_gt_label_valid(self) -> None:
        """Test validation of valid ground truth labels."""
        labels = torch.tensor([0, 1])
        validated_labels = self.validator.validate_gt_label(labels)
        assert isinstance(validated_labels, torch.Tensor)
        assert validated_labels.dtype == torch.bool
        assert torch.equal(validated_labels, torch.tensor([False, True]))

    def test_validate_gt_label_none(self) -> None:
        """Test validation of None ground truth labels."""
        assert self.validator.validate_gt_label(None) is None

    def test_validate_gt_label_invalid_type(self) -> None:
        """Test validation of ground truth labels with invalid type."""
        with pytest.raises(TypeError, match="Ground truth labels must be a torch.Tensor"):
            self.validator.validate_gt_label(["0", "1"])

    def test_validate_gt_mask_valid(self) -> None:
        """Test validation of valid ground truth masks."""
        masks = torch.randint(0, 2, (10, 1, 224, 224))
        validated_masks = self.validator.validate_gt_mask(masks)
        assert isinstance(validated_masks, Mask)
        assert validated_masks.shape == (10, 224, 224)
        assert validated_masks.dtype == torch.bool

    def test_validate_gt_mask_none(self) -> None:
        """Test validation of None ground truth masks."""
        assert self.validator.validate_gt_mask(None) is None

    def test_validate_gt_mask_invalid_type(self) -> None:
        """Test validation of ground truth masks with invalid type."""
        with pytest.raises(TypeError, match="Ground truth mask must be a torch.Tensor"):
            self.validator.validate_gt_mask([torch.zeros(10, 224, 224)])

    def test_validate_gt_mask_invalid_shape(self) -> None:
        """Test validation of ground truth masks with invalid shape."""
        with pytest.raises(ValueError, match="Ground truth mask must have 1 channel, got 2."):
            self.validator.validate_gt_mask(torch.zeros(10, 2, 224, 224))

    def test_validate_anomaly_map_valid(self) -> None:
        """Test validation of a valid anomaly map batch."""
        anomaly_map = torch.rand(2, 10, 224, 224)
        validated_map = self.validator.validate_anomaly_map(anomaly_map)
        assert isinstance(validated_map, Mask)
        assert validated_map.shape == (2, 10, 224, 224)
        assert validated_map.dtype == torch.float32

    def test_validate_anomaly_map_none(self) -> None:
        """Test validation of a None anomaly map batch."""
        assert self.validator.validate_anomaly_map(None) is None

    def test_validate_anomaly_map_invalid_shape(self) -> None:
        """Test validation of an anomaly map batch with invalid shape."""
        with pytest.raises(ValueError, match="Anomaly maps must have 1 channel, got 2."):
            self.validator.validate_anomaly_map(torch.rand(2, 10, 2, 224, 224))

    def test_validate_pred_score_valid(self) -> None:
        """Test validation of valid prediction scores."""
        scores = torch.tensor([0.1, 0.2])
        validated_scores = self.validator.validate_pred_score(scores)
        assert torch.equal(validated_scores, scores)

    def test_validate_pred_score_none_with_anomaly_map(self) -> None:
        """Test validation of None prediction scores with anomaly map."""
        anomaly_map = torch.rand(2, 10, 224, 224)
        computed_scores = self.validator.validate_pred_score(None, anomaly_map)
        assert computed_scores.shape == (2,)

    def test_validate_pred_label_valid(self) -> None:
        """Test validation of valid prediction labels."""
        labels = torch.tensor([1, 0])
        validated_labels = self.validator.validate_pred_label(labels)
        assert torch.equal(validated_labels, torch.tensor([True, False]))

    def test_validate_pred_label_none(self) -> None:
        """Test validation of None prediction labels."""
        assert self.validator.validate_pred_label(None) is None

    def test_validate_pred_label_invalid_shape(self) -> None:
        """Test validation of prediction labels with invalid shape."""
        with pytest.raises(ValueError, match="Predicted labels must be a 1D tensor"):
            self.validator.validate_pred_label(torch.tensor([[1], [0]]))
