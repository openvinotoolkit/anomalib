"""Unit tests for TimmFeatureExtractor."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import torch
from timm.models import create_model

from anomalib.models.components.feature_extractors.timm import TimmFeatureExtractor


def test_backbone_weight_file() -> None:
    """Test the backbone weight file loading mechanism."""
    # Use the simplest model.
    backbone = "resnet18"
    _, state_dict_fpath = tempfile.mkstemp()
    # Only examine conv1 before layers in the feature extractor.
    layers = []

    # Get random model weights without downloading.
    model = create_model(backbone, pretrained=False)
    state_dict = model.state_dict()

    # Set the conv1 weight to zero, and save state_dict to a temp file.
    state_dict["conv1.weight"].zero_()
    torch.save(state_dict, state_dict_fpath)

    # Load weights from the temp file.
    backbone_with_path = f"{backbone}__AT__{state_dict_fpath}"

    fe_restored = TimmFeatureExtractor(backbone_with_path, layers, pre_trained=True)
    Path(state_dict_fpath).unlink()
    # The weights should be zero if the file loading mechanism works.
    assert torch.all(fe_restored.feature_extractor.conv1.weight == 0)
