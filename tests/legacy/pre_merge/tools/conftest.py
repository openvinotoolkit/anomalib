"""Fixtures for the tools tests."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Generator

import albumentations as A  # noqa: N812
import cv2
import numpy as np
import pytest
from albumentations.pytorch import ToTensorV2


@pytest.fixture(scope="package")
def get_dummy_inference_image(project_path: Path) -> str:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(project_path) + "/dummy_image.png", image)
    return str(project_path) + "/dummy_image.png"


@pytest.fixture(scope="package")
def transforms_config() -> dict:
    """Note: this is computed using trainer.datamodule.test_data.transform.to_dict()"""
    return A.Compose([A.ToFloat(max_value=255), ToTensorV2()]).to_dict()
