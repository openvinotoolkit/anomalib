"""Unit tests for the WinCLIP torch model."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from anomalib.models.image.winclip.torch_model import WinClipModel


class TestSetupWinClipModel:
    """Test the WinCLIP torch model."""

    def test_zero_shot_from_init(self) -> None:
        """Test WinCLIP initialization from init method in zero-shot mode."""
        model = WinClipModel(class_name="item")
        assert model.k_shot == 0
        assert getattr(model, "text_embeddings", None) is not None

    def test_zero_shot_from_setup(self) -> None:
        """Test WinCLIP initialization from setup method in zero-shot mode."""
        model = WinClipModel()
        model.setup(class_name="item")
        assert model.k_shot == 0
        assert getattr(model, "text_embeddings", None) is not None

    @pytest.mark.parametrize("apply_transform", [True, False])
    def test_few_shot_from_init(self, apply_transform: bool) -> None:
        """Test WinCLIP initialization from init in few-shot mode."""
        ref_images = torch.rand(2, 3, 240, 240)
        model = WinClipModel(class_name="item", reference_images=ref_images, apply_transform=apply_transform)
        assert model.k_shot == 2
        assert getattr(model, "text_embeddings", None) is not None
        assert getattr(model, "visual_embeddings", None) is not None

    @pytest.mark.parametrize("apply_transform", [True, False])
    def test_few_shot_from_setup(self, apply_transform: bool) -> None:
        """Test WinCLIP initialization from setup method in few-shot mode."""
        ref_images = torch.rand(2, 3, 240, 240)
        model = WinClipModel(apply_transform=apply_transform)
        model.setup(class_name="item", reference_images=ref_images)
        assert model.k_shot == 2
        assert getattr(model, "text_embeddings", None) is not None
        assert getattr(model, "visual_embeddings", None) is not None

    def test_raises_error_when_not_initialized(self) -> None:
        """Test if an error is raised when trying to access un-initialized attributes."""
        model = WinClipModel()
        with pytest.raises(RuntimeError):
            _ = model.text_embeddings
        with pytest.raises(RuntimeError):
            _ = model.visual_embeddings
        with pytest.raises(RuntimeError):
            _ = model.patch_embeddings
