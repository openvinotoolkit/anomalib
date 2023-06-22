"""Fixtures for the tools tests."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from tempfile import TemporaryDirectory
from typing import Generator

import albumentations as A
import cv2
import numpy as np
import pytest
from albumentations.pytorch import ToTensorV2

from anomalib.config import get_configurable_parameters


@pytest.fixture(scope="package")
def project_path() -> Generator[str, None, None]:
    with TemporaryDirectory() as project_dir:
        yield project_dir


@pytest.fixture(scope="package")
def get_config(project_path):
    def get_config(
        model_name: str | None = None,
        config_path: str | None = None,
        weight_file: str | None = None,
    ):
        """Gets config for testing."""
        config = get_configurable_parameters(model_name, config_path, weight_file)
        config.dataset.image_size = (100, 100)
        config.model.input_size = (100, 100)
        config.project.path = project_path
        config.trainer.max_epochs = 1
        config.trainer.check_val_every_n_epoch = 1
        config.trainer.limit_train_batches = 1
        config.trainer.limit_predict_batches = 1
        return config

    yield get_config


@pytest.fixture(scope="package")
def get_dummy_inference_image(project_path) -> Generator[str, None, None]:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(project_path + "/dummy_image.png", image)
    yield project_path + "/dummy_image.png"


@pytest.fixture(scope="package")
def transforms_config() -> dict:
    """Note: this is computed using trainer.datamodule.test_data.transform.to_dict()"""
    return A.Compose([A.ToFloat(max_value=255), ToTensorV2()]).to_dict()
