"""Test API.

Tests the models using API. The weight paths from the trained models are used for the rest of the tests.
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import pytest
from lightning.pytorch.callbacks import ModelCheckpoint

from anomalib.data import AnomalibDataModule, MVTec, TaskType, UCSDped
from anomalib.engine import Engine
from anomalib.models import AnomalyModule, get_available_models, get_model


class TestAPI:
    """Do sanity check on all models."""

    def _get_objects(
        self,
        model_name: str,
        dataset_path: Path,
        project_path: Path,
    ) -> tuple[AnomalyModule, AnomalibDataModule, Engine]:
        """Return model, dataset, and engine objects.

        Args:
            model_name (str): Name of the model to train
            dataset_path (Path): Path to the root of dummy dataset
            project_path (Path): path to the temporary project folder

        Returns:
            tuple[AnomalyModule, AnomalibDataModule, Engine]: _description_
        """
        # select task type
        if model_name in ("rkde", "ai_vad"):
            task_type = TaskType.DETECTION
        elif model_name in ("ganomaly", "dfkde"):
            task_type = TaskType.CLASSIFICATION
        else:
            task_type = TaskType.SEGMENTATION

        # select dataset image size
        image_size = (224, 224) if model_name == "patchcore" else (256, 256)

        # select dataset
        if model_name == "ai_vad":
            # aivad expects UCSD dataset
            dataset = UCSDped(root=dataset_path / "ucsdped", category="dummy", task=task_type)
        else:
            dataset = MVTec(root=dataset_path / "mvtec", category="dummy", image_size=image_size, task=task_type)

        model = get_model(model_name)
        engine = Engine(
            logger=False,
            default_root_dir=project_path,
            max_epochs=1,
            check_val_every_n_epoch=1,
            devices=1,
            pixel_metrics=["F1Score", "AUROC"],
            task=task_type,
            callbacks=[
                ModelCheckpoint(
                    dirpath=project_path / f"{model_name}/dummy/weights",
                    monitor=None,
                    filename="last",
                    save_last=True,
                    auto_insert_metric_name=False,
                ),
            ],
        )
        return model, dataset, engine

    @pytest.mark.parametrize("model_name", get_available_models())
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

    @pytest.mark.parametrize("model_name", get_available_models())
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
        engine.test(model=model, datamodule=dataset, ckpt_path=f"{project_path}/{model_name}/dummy/weights/last.ckpt")
