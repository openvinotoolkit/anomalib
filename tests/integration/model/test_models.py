"""Test API.

Tests the models using API. The weight paths from the trained models are used for the rest of the tests.
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import contextlib
import sys
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from anomalib.data import AnomalibDataModule, MVTecAD
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import AnomalibModule, get_available_models, get_model


def models() -> set[str]:
    """Return all available models."""
    return get_available_models()


def export_types() -> list[ExportType]:
    """Return all available export frameworks."""
    return list(ExportType)


@contextlib.contextmanager
def increased_recursion_limit(limit: int = 10000) -> Generator[None, None, None]:
    """Temporarily increase the recursion limit."""
    old_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(limit)
        yield
    finally:
        sys.setrecursionlimit(old_limit)


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
        model, dataset, engine = self._get_objects(
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
        model, dataset, engine = self._get_objects(
            model_name=model_name,
            dataset_path=dataset_path,
            project_path=project_path,
        )
        engine.test(
            model=model,
            datamodule=dataset,
            ckpt_path=f"{project_path}/{model.name}/{dataset.name}/dummy/v0/weights/lightning/model.ckpt",
        )

    @pytest.mark.parametrize("model_name", models())
    def test_train(self, model_name: str, dataset_path: Path, project_path: Path) -> None:
        """Train model from checkpoint.

        Args:
            model_name (str): Name of the model.
            dataset_path (Path): Root to dataset from fixture.
            project_path (Path): Path to temporary project folder from fixture.
        """
        model, dataset, engine = self._get_objects(
            model_name=model_name,
            dataset_path=dataset_path,
            project_path=project_path,
        )
        engine.train(
            model=model,
            datamodule=dataset,
            ckpt_path=f"{project_path}/{model.name}/{dataset.name}/dummy/v0/weights/lightning/model.ckpt",
        )

    @pytest.mark.parametrize("model_name", models())
    def test_validate(self, model_name: str, dataset_path: Path, project_path: Path) -> None:
        """Validate model from checkpoint.

        Args:
            model_name (str): Name of the model.
            dataset_path (Path): Root to dataset from fixture.
            project_path (Path): Path to temporary project folder from fixture.
        """
        model, dataset, engine = self._get_objects(
            model_name=model_name,
            dataset_path=dataset_path,
            project_path=project_path,
        )
        engine.validate(
            model=model,
            datamodule=dataset,
            ckpt_path=f"{project_path}/{model.name}/{dataset.name}/dummy/v0/weights/lightning/model.ckpt",
        )

    @pytest.mark.parametrize("model_name", models())
    def test_predict(self, model_name: str, dataset_path: Path, project_path: Path) -> None:
        """Predict using model from checkpoint.

        Args:
            model_name (str): Name of the model.
            dataset_path (Path): Root to dataset from fixture.
            project_path (Path): Path to temporary project folder from fixture.
        """
        model, datamodule, engine = self._get_objects(
            model_name=model_name,
            dataset_path=dataset_path,
            project_path=project_path,
        )
        engine.predict(
            model=model,
            ckpt_path=f"{project_path}/{model.name}/{datamodule.name}/dummy/v0/weights/lightning/model.ckpt",
            datamodule=datamodule,
        )

    @pytest.mark.parametrize("model_name", models())
    @pytest.mark.parametrize("export_type", export_types())
    def test_export(
        self,
        model_name: str,
        export_type: ExportType,
        dataset_path: Path,
        project_path: Path,
    ) -> None:
        """Export model from checkpoint.

        Args:
            model_name (str): Name of the model.
            export_type (ExportType): Framework to export to.
            dataset_path (Path): Root to dataset from fixture.
            project_path (Path): Path to temporary project folder from fixture.
        """
        model, dataset, engine = self._get_objects(
            model_name=model_name,
            dataset_path=dataset_path,
            project_path=project_path,
        )

        # Use context manager only for CSFlow
        with increased_recursion_limit() if model_name == "csflow" else contextlib.nullcontext():
            engine.export(
                model=model,
                ckpt_path=f"{project_path}/{model.name}/{dataset.name}/dummy/v0/weights/lightning/model.ckpt",
                export_type=export_type,
            )

    @staticmethod
    def _get_objects(
        model_name: str,
        dataset_path: Path,
        project_path: Path,
    ) -> tuple[AnomalibModule, AnomalibDataModule, Engine]:
        """Return model, dataset, and engine objects.

        Args:
            model_name (str): Name of the model to train
            dataset_path (Path): Path to the root of dummy dataset
            project_path (Path): path to the temporary project folder

        Returns:
            tuple[AnomalibModule, AnomalibDataModule, Engine]: Returns the created objects for model, dataset,
                and engine
        """
        # set extra model args
        # TODO(ashwinvaidya17): Fix these Edge cases
        # https://github.com/openvinotoolkit/anomalib/issues/1478

        extra_args = {}
        if model_name == "dfkde":
            extra_args["n_pca_components"] = 2

        if model_name == {"ai_vad", "fuvas"}:
            pytest.skip("Revisit video models tests")
        else:
            # EfficientAd requires that the batch size be lesser than the number of images in the dataset.
            # This is so that the LR step size is not 0.
            dataset = MVTecAD(
                root=dataset_path / "mvtecad",
                category="dummy",
                # EfficientAd requires train batch size 1
                train_batch_size=1 if model_name == "efficient_ad" else 2,
            )

        model = get_model(model_name, **extra_args)

        if model_name == "vlm_ad":
            model.vlm_backend = MagicMock()
            model.vlm_backend.predict.return_value = "YES: Because reasons..."

        engine = Engine(
            logger=False,
            default_root_dir=project_path,
            max_epochs=1,
            devices=1,
            # TODO(ashwinvaidya17): Fix these Edge cases
            # https://github.com/openvinotoolkit/anomalib/issues/1478
            max_steps=70000 if model_name == "efficient_ad" else -1,
        )
        return model, dataset, engine
