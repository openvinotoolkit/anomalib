"""Test AnomalyModule module."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from anomalib.models.components.base import AnomalyModule


@pytest.fixture(scope="class")
def model_config_folder_path() -> str:
    """Fixture that returns model config folder path."""
    return "configs/model"


class TestAnomalyModule:
    """Test AnomalyModule."""

    @pytest.fixture(autouse=True)
    def setup(self, model_config_folder_path: str) -> None:
        """Setup test AnomalyModule."""
        self.model_config_folder_path = model_config_folder_path

    @staticmethod
    def test_from_config_with_wrong_config_path() -> None:
        """Test AnomalyModule.from_config with wrong model name."""
        with pytest.raises(FileNotFoundError):
            AnomalyModule.from_config(config_path="wrong_configs.yaml")

    @pytest.mark.parametrize(
        "model_name",
        [
            "ai_vad",
            "cfa",
            "cflow",
            "csflow",
            "dfkde",
            "dfm",
            "draem",
            "dsr",
            "efficient_ad",
            "fastflow",
            "ganomaly",
            "padim",
            "patchcore",
            "reverse_distillation",
            "rkde",
            "stfpm",
            "uflow",
        ],
    )
    def test_from_config(self, model_name: str) -> None:
        """Test AnomalyModule.from_config."""
        config_path = Path(self.model_config_folder_path) / f"{model_name}.yaml"
        model = AnomalyModule.from_config(config_path=config_path)
        assert model is not None
        assert isinstance(model, AnomalyModule)
