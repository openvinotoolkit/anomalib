"""Test CLI entrypoints.

This just checks if one of the model works end-to-end. The rest of the models are checked using the API.
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import pytest
import torch

from anomalib.cli import AnomalibCLI
from anomalib.data import MVTec, UCSDped
from anomalib.deploy.export import ExportType
from anomalib.utils.types import TaskType


@pytest.fixture(scope="module")
def model_name() -> str:
    """Return the name of the model used throughout the CLI tests."""
    return "Padim"


class TestCLI:
    """Do sanity check on all models."""

    def test_fit(self, model_name: str, dataset_path: Path, project_path: Path) -> None:
        """Test fit CLI.

        Args:
            model_name: Name of the model to test.
            dataset_path (Path): Root of the synthetic/original dataset.
            project_path (Path): Path to temporary project folder.
        """
        AnomalibCLI(
            args=[
                "fit",
                "--model",
                model_name,
                *self._get_common_cli_args(model_name, dataset_path, project_path),
                # TODO(ashwinvaidya17): Fix these Edge cases
                # https://github.com/openvinotoolkit/anomalib/issues/1478
                "--data.train_batch_size",
                "2",  # batch size is 2 so that len train_dataloader does not return 1 in EfficientAd
            ],
        )
        torch.cuda.empty_cache()

    def test_test(self, model_name: str, dataset_path: Path, project_path: Path) -> None:
        """Test the test method of the CLI.

        Args:
            model_name: Name of the model to test.
            dataset_path (Path): Root of the synthetic/original dataset.
            project_path (Path): Path to temporary project folder.
        """
        AnomalibCLI(
            args=[
                "test",
                "--model",
                model_name,
                *self._get_common_cli_args(model_name, dataset_path, project_path),
                "--ckpt_path",
                f"{project_path}/{model_name}/dummy/weights/last.ckpt",
            ],
        )
        torch.cuda.empty_cache()

    def test_train(self, model_name: str, dataset_path: Path, project_path: Path) -> None:
        """Test the train method of the CLI.

        Args:
            model_name: Name of the model to test.
            dataset_path (Path): Root of the synthetic/original dataset.
            project_path (Path): Path to temporary project folder.
        """
        AnomalibCLI(
            args=[
                "train",
                "--model",
                model_name,
                *self._get_common_cli_args(model_name, dataset_path, project_path),
                "--ckpt_path",
                f"{project_path}/{model_name}/dummy/weights/last.ckpt",
            ],
        )
        torch.cuda.empty_cache()

    def test_validate(self, model_name: str, dataset_path: Path, project_path: Path) -> None:
        """Test the validate method of the CLI.

        Args:
            model_name: Name of the model to test.
            dataset_path (Path): Root of the synthetic/original dataset.
            project_path (Path): Path to temporary project folder.
        """
        AnomalibCLI(
            args=[
                "validate",
                "--model",
                model_name,
                *self._get_common_cli_args(model_name, dataset_path, project_path),
                "--ckpt_path",
                f"{project_path}/{model_name}/dummy/weights/last.ckpt",
            ],
        )
        torch.cuda.empty_cache()

    def test_predict(self, model_name: str, dataset_path: Path, project_path: Path) -> None:
        """Test the predict method of the CLI.

        Args:
            model_name: Name of the model to test.
            dataset_path (Path): Root of the synthetic/original dataset.
            project_path (Path): Path to temporary project folder.
        """
        AnomalibCLI(
            args=[
                "predict",
                "--model",
                model_name,
                *self._get_common_cli_args(
                    model_name,
                    dataset_path,
                    project_path,
                ),
                "--ckpt_path",
                f"{project_path}/{model_name}/dummy/weights/last.ckpt",
            ],
        )
        torch.cuda.empty_cache()

    @pytest.mark.parametrize("export_type", [ExportType.TORCH, ExportType.ONNX, ExportType.OPENVINO])
    def test_export(
        self,
        model_name: str,
        dataset_path: Path,
        project_path: Path,
        export_type: ExportType,
    ) -> None:
        """Test the export method of the CLI.

        Args:
            model_name: Name of the model to test.
            dataset_path (Path): Root of the synthetic/original dataset.
            project_path (Path): Path to temporary project folder.
            export_type (ExportType): Export type.
        """
        AnomalibCLI(
            args=[
                "export",
                "--model",
                model_name,
                "--export_type",
                export_type,
                *self._get_common_cli_args(model_name, dataset_path, project_path),
                "--input_size",
                "[256, 256]",
            ],
        )

    @staticmethod
    def _get_common_cli_args(model_name: str, dataset_path: Path, project_path: Path) -> list[str]:
        """Return common CLI args for all models.

        Args:
            model_name (str): Name of the model class.
            dataset_path (Path): Path to the dataset.
            project_path (Path): Path to the project folder.
        """
        # We need to set the predict dataloader as MVTec and UCSDped do have have predict_dataloader attribute defined.
        if model_name == "AiVad":
            data_root = f"{dataset_path}/ucsdped"
            dataclass = "UCSDped"
            UCSDped.predict_dataloader = UCSDped.test_dataloader
        else:
            data_root = f"{dataset_path}/mvtec"
            dataclass = "MVTec"
            MVTec.predict_dataloader = MVTec.test_dataloader

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
