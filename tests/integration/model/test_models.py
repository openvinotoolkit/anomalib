"""Test API.

Tests the models using API. The weight paths from the trained models are used for the rest of the tests.
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import pytest

from anomalib.models import get_available_models
from tests.integration.model.conftest import get_objects


def models() -> set[str]:
    """Return all available models except ai_vad."""
    return get_available_models()


class TestAPI:
    """Do sanity check on all models."""

    @pytest.mark.parametrize("model_name", models())
    def test_fit(self, model_name: str, dataset_path: Path, project_path: Path) -> None:
        """Fit the model and save checkpoint.

        Args:
            model_name (str): Name of the model.
            dataset_path (Path): Root to dataset from fixture.
            project_path (Path): Path to temporary project folder from fixture.
        """
        model, dataset, engine = get_objects(
            model_name=model_name,
            dataset_path=dataset_path,
            project_path=project_path,
        )
        engine.fit(model=model, datamodule=dataset)

    @pytest.mark.parametrize("model_name", models())
    def test_test(self, model_name: str, dataset_path: Path, project_path: Path) -> None:
        """Test model from checkpoint.

        Args:
            model_name (str): Name of the model.
            dataset_path (Path): Root to dataset from fixture.
            project_path (Path): Path to temporary project folder from fixture.
        """
        model, dataset, engine = get_objects(
            model_name=model_name,
            dataset_path=dataset_path,
            project_path=project_path,
        )
        engine.test(model=model, datamodule=dataset, ckpt_path=f"{project_path}/{model_name}/dummy/weights/last.ckpt")
