"""Test CLI entrypoints.

This just checks if one of the model works end-to-end. The rest of the models are checked using the API.
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import random
from pathlib import Path

import pytest
import torch

from anomalib.data import TaskType
from anomalib.models import AnomalyModule
from anomalib.utils.cli import AnomalibCLI


@pytest.fixture(scope="module")
def random_model_name() -> str:
    """Return a random model name."""
    # TODO(ashwinvaidya17): Restore AiVad test
    # CVS-109972
    all_models = [model.__name__ for model in AnomalyModule.__subclasses__() if model.__name__ != "AiVad"]
    return random.choice(all_models)  # noqa: S311


class TestCLI:
    """Do sanity check on all models."""

    def test_fit(self, random_model_name: str, dataset_path: Path, project_path: Path) -> None:
        """Test fit CLI.

        Args:
            random_model_name: Name of the model to test.
            dataset_path (Path): Root of the synthetic/original dataset.
            project_path (Path): Path to temporary project folder.
        """
        AnomalibCLI(
            args=[
                "fit",
                "--model",
                random_model_name,
                *self._get_common_cli_args(random_model_name, dataset_path, project_path),
                # TODO(ashwinvaidya17): Fix these Edge cases
                # https://github.com/openvinotoolkit/anomalib/issues/1478
                "--data.train_batch_size",
                "2",  # batch size is 2 so that len train_dataloader does not return 1 in EfficientAd
            ],
        )
        torch.cuda.empty_cache()

    def test_test(self, random_model_name: str, dataset_path: Path, project_path: Path) -> None:
        """Test the test method of the CLI.

        Args:
            random_model_name: Name of the model to test.
            dataset_path (Path): Root of the synthetic/original dataset.
            project_path (Path): Path to temporary project folder.
        """
        AnomalibCLI(
            args=[
                "test",
                "--model",
                random_model_name,
                *self._get_common_cli_args(random_model_name, dataset_path, project_path),
                "--ckpt_path",
                f"{project_path}/{random_model_name}/dummy/weights/last.ckpt",
            ],
        )
        torch.cuda.empty_cache()

        # TODO(ashwinvaidya17): Validate
        # CVS-109972

        # TODO(ashwinvaidya17): Predict
        # CVS-109972

        # TODO(ashwinvaidya17): export
        # CVS-109972

    @staticmethod
    def _get_common_cli_args(model_name: str, dataset_path: Path, project_path: Path) -> list[str]:
        """Return common CLI args for all models.

        Args:
            model_name (str): Name of the model class.
            dataset_path (Path): Path to the dataset.
            project_path (Path): Path to the project folder.
        """
        # batch size of 8 is taken so that the lr computation for efficient_ad does not return 0 when
        # max_epochs is 1
        if model_name == "AiVad":
            data_root = f"{dataset_path}/ucsdped"
            dataclass = "UCSDped"
        else:
            data_root = f"{dataset_path}/mvtec"
            dataclass = "MVTec"

        task_type = TaskType.SEGMENTATION
        if model_name in ("Rkde", "AiVad"):
            task_type = TaskType.DETECTION
        elif model_name in ("Dfkde", "Ganomaly"):
            task_type = TaskType.CLASSIFICATION

        # TODO(ashwinvaidya17): Fix these Edge cases
        # https://github.com/openvinotoolkit/anomalib/issues/1478
        extra_args = []
        if model_name in ("Rkde", "Dfkde"):
            # since dataset size is smaller than the number of default pca  components
            extra_args = ["--model.n_pca_components", "2"]
        elif model_name == "EfficientAd":
            # max steps need to be set for lr scheduler
            # but with this set, max_epochs * len(train_dataloader) has a small value
            extra_args = ["--trainer.max_steps", "70000"]

        return [
            *extra_args,
            "--data",
            dataclass,
            "--data.root",
            data_root,
            "--data.category",
            "dummy",
            "--results_dir.path",
            str(project_path),
            "--results_dir.unique",
            "false",
            "--task",
            task_type,
            "--trainer.max_epochs",
            "1",
            "--trainer.callbacks+=ModelCheckpoint",
            "--trainer.callbacks.dirpath",
            f"{project_path}/{model_name}/dummy/weights",
            "--trainer.callbacks.monitor",
            "null",
            "--trainer.callbacks.filename",
            "last",
            "--trainer.callbacks.save_last",
            "true",
            "--trainer.callbacks.auto_insert_metric_name",
            "false",
        ]
