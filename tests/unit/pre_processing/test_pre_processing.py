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
