"""Test the PreProcessor class."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage
from torchvision.tv_tensors import Image, Mask

from anomalib.data import ImageBatch
from anomalib.pre_processing import PreProcessor


class TestPreProcessor:
    """Test the PreProcessor class."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test fixtures for each test method."""
        image = Image(torch.rand(3, 256, 256))
        gt_mask = Mask(torch.zeros(256, 256))
        self.dummy_batch = ImageBatch(image=image, gt_mask=gt_mask)
        self.common_transform = Compose([Resize((224, 224)), ToImage(), ToDtype(torch.float32, scale=True)])

    def test_init(self) -> None:
        """Test the initialization of the PreProcessor class."""
        # Test with stage-specific transforms
        train_transform = Compose([Resize((224, 224)), ToImage(), ToDtype(torch.float32, scale=True)])
        val_transform = Compose([Resize((256, 256)), ToImage(), ToDtype(torch.float32, scale=True)])
        pre_processor = PreProcessor(train_transform=train_transform, val_transform=val_transform)
        assert pre_processor.train_transform == train_transform
        assert pre_processor.val_transform == val_transform
        assert pre_processor.test_transform is None

        # Test with single transform for all stages
        pre_processor = PreProcessor(transform=self.common_transform)
        assert pre_processor.train_transform == self.common_transform
        assert pre_processor.val_transform == self.common_transform
        assert pre_processor.test_transform == self.common_transform

        # Test error case: both transform and stage-specific transform
        with pytest.raises(ValueError, match="`transforms` cannot be used together with"):
            PreProcessor(transform=self.common_transform, train_transform=train_transform)

    def test_forward(self) -> None:
        """Test the forward method of the PreProcessor class."""
        pre_processor = PreProcessor(transform=self.common_transform)
        processed_batch = pre_processor(self.dummy_batch.image)
        assert isinstance(processed_batch, torch.Tensor)
        assert processed_batch.shape == (1, 3, 224, 224)

    def test_no_transform(self) -> None:
        """Test no transform."""
        pre_processor = PreProcessor()
        processed_batch = pre_processor(self.dummy_batch.image)
        assert isinstance(processed_batch, torch.Tensor)
        assert processed_batch.shape == (1, 3, 256, 256)

    @staticmethod
    def test_different_stage_transforms() -> None:
        """Test different stage transforms."""
        train_transform = Compose([Resize((224, 224)), ToImage(), ToDtype(torch.float32, scale=True)])
        val_transform = Compose([Resize((256, 256)), ToImage(), ToDtype(torch.float32, scale=True)])
        test_transform = Compose([Resize((288, 288)), ToImage(), ToDtype(torch.float32, scale=True)])

        pre_processor = PreProcessor(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
        )

        # Test train transform
        test_batch = ImageBatch(image=Image(torch.rand(3, 256, 256)), gt_mask=Mask(torch.zeros(256, 256)))
        processed_batch = pre_processor.train_transform(test_batch.image)
        assert isinstance(processed_batch, torch.Tensor)
        assert processed_batch.shape == (1, 3, 224, 224)

        # Test validation transform
        test_batch = ImageBatch(image=Image(torch.rand(3, 256, 256)), gt_mask=Mask(torch.zeros(256, 256)))
        processed_batch = pre_processor.val_transform(test_batch.image)
        assert isinstance(processed_batch, torch.Tensor)
        assert processed_batch.shape == (1, 3, 256, 256)

        # Test test transform
        test_batch = ImageBatch(image=Image(torch.rand(3, 256, 256)), gt_mask=Mask(torch.zeros(256, 256)))
        processed_batch = pre_processor.test_transform(test_batch.image)
        assert isinstance(processed_batch, torch.Tensor)
        assert processed_batch.shape == (1, 3, 288, 288)
