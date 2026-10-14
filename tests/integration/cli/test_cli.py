# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Test CLI entrypoints on Padim model.

This just checks if one of the model works end-to-end. The rest of the models are checked using the API.
"""

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from anomalib.cli import AnomalibCLI
from anomalib.deploy import ExportType


class TestCLI:
    """Do sanity check on all models."""

    @pytest.fixture(scope="class", autouse=True)
    @staticmethod
    def _make_datasets(make_dummy_dataset: Callable[[str], Path]) -> None:
        """Generate datasets used by CLI data and image-path tests."""
        make_dummy_dataset("mvtecad")
        make_dummy_dataset("visa")

    def test_fit(self, dataset_path: Path, project_path: Path) -> None:
        """Test fit CLI.

        Args:
            dataset_path (Path): Root of the synthetic/original dataset.
            project_path (Path): Path to temporary project folder.
        """
        AnomalibCLI(
            args=[
                "fit",
                *self._get_common_cli_args(dataset_path, project_path),
            ],
        )
        torch.cuda.empty_cache()

    def test_test(
        self,
        dataset_path: Path,
        project_path: Path,
        ckpt_path: Callable[[str], Path],
    ) -> None:
        """Test the test method of the CLI.

        Args:
            dataset_path (Path): Root of the synthetic/original dataset.
            project_path (Path): Path to temporary project folder.
            ckpt_path (Callable[[str], Path]): Resolve a shared model checkpoint.
        """
        padim_ckpt = ckpt_path("Padim")
        AnomalibCLI(
            args=[
                "test",
                *self._get_common_cli_args(dataset_path, project_path),
                "--ckpt_path",
                str(padim_ckpt),
            ],
        )
        torch.cuda.empty_cache()

    def test_train(
        self,
        dataset_path: Path,
        project_path: Path,
        ckpt_path: Callable[[str], Path],
    ) -> None:
        """Test the train method of the CLI.

        Args:
            dataset_path (Path): Root of the synthetic/original dataset.
            project_path (Path): Path to temporary project folder.
            ckpt_path (Callable[[str], Path]): Resolve a shared model checkpoint.
        """
        padim_ckpt = ckpt_path("Padim")
        AnomalibCLI(
            args=[
                "train",
                *self._get_common_cli_args(dataset_path, project_path),
                "--ckpt_path",
                str(padim_ckpt),
            ],
        )
        torch.cuda.empty_cache()

    def test_validate(
        self,
        dataset_path: Path,
        project_path: Path,
        ckpt_path: Callable[[str], Path],
    ) -> None:
        """Test the validate method of the CLI.

        Args:
            dataset_path (Path): Root of the synthetic/original dataset.
            project_path (Path): Path to temporary project folder.
            ckpt_path (Callable[[str], Path]): Resolve a shared model checkpoint.
        """
        padim_ckpt = ckpt_path("Padim")
        AnomalibCLI(
            args=[
                "validate",
                *self._get_common_cli_args(dataset_path, project_path),
                "--ckpt_path",
                str(padim_ckpt),
            ],
        )
        torch.cuda.empty_cache()

    def test_predict_with_dataloader(
        self,
        dataset_path: Path,
        project_path: Path,
        ckpt_path: Callable[[str], Path],
    ) -> None:
        """Test the predict method of the CLI.

        This test uses the MVTec AD dataloader for predict test.

        Args:
            dataset_path (Path): Root of the synthetic/original dataset.
            project_path (Path): Path to temporary project folder.
            ckpt_path (Callable[[str], Path]): Resolve a shared model checkpoint.
        """
        padim_ckpt = ckpt_path("Padim")
        # Test with MVTec AD Dataset
        AnomalibCLI(
            args=[
                "predict",
                *self._get_common_cli_args(
                    dataset_path,
                    project_path,
                ),
                "--ckpt_path",
                str(padim_ckpt),
            ],
        )
        torch.cuda.empty_cache()

    def test_predict_with_image_folder(
        self,
        project_path: Path,
        ckpt_path: Callable[[str], Path],
    ) -> None:
        """Test the predict method of the CLI.

        This test uses the path to image folder for predict test.

        Args:
            project_path (Path): Path to temporary project folder.
            ckpt_path (Callable[[str], Path]): Resolve a shared model checkpoint.
        """
        padim_ckpt = ckpt_path("Padim")
        # Test with image path
        AnomalibCLI(
            args=[
                "predict",
                "--data",
                f"{project_path}/datasets/visa_pytorch/dummy/test/bad",
                *self._get_common_cli_args(
                    None,
                    project_path,
                ),
                "--ckpt_path",
                str(padim_ckpt),
            ],
        )
        torch.cuda.empty_cache()

    def test_predict_with_image_path(
        self,
        project_path: Path,
        ckpt_path: Callable[[str], Path],
    ) -> None:
        """Test the predict method of the CLI.

        This test uses the path to image for predict test.

        Args:
            project_path (Path): Path to temporary project folder.
            ckpt_path (Callable[[str], Path]): Resolve a shared model checkpoint.
        """
        padim_ckpt = ckpt_path("Padim")
        # Test with image path
        AnomalibCLI(
            args=[
                "predict",
                "--data",
                f"{project_path}/datasets/visa_pytorch/dummy/test/bad/000.jpg",
                *self._get_common_cli_args(
                    None,
                    project_path,
                ),
                "--ckpt_path",
                str(padim_ckpt),
            ],
        )
        torch.cuda.empty_cache()

    @pytest.mark.parametrize("export_type", [ExportType.TORCH, ExportType.ONNX, ExportType.OPENVINO])
    def test_export(
        self,
        project_path: Path,
        export_type: ExportType,
        ckpt_path: Callable[[str], Path],
    ) -> None:
        """Test the export method of the CLI.

        Args:
            project_path (Path): Path to temporary project folder.
            export_type (ExportType): Export type.
            ckpt_path (Callable[[str], Path]): Resolve a shared model checkpoint.
        """
        padim_ckpt = ckpt_path("Padim")
        AnomalibCLI(
            args=[
                "export",
                "--export_type",
                export_type,
                *self._get_common_cli_args(None, project_path),
                "--ckpt_path",
                str(padim_ckpt),
            ],
        )

    @staticmethod
    def _get_common_cli_args(dataset_path: Path | None, project_path: Path) -> list[str]:
        """Return common CLI args for all models.

        Args:
            dataset_path (Path): Path to the dataset. If the path is None, data arguments are not appended to the args.
            project_path (Path): Path to the project folder.
            model_name (str): Name of the model. Defaults to None.
        """
        # We need to set the predict dataloader as MVTec AD and UCSDped do not
        # have predict_dataloader attribute defined.
        if dataset_path:
            data_root = f"{dataset_path}/mvtecad"
            dataclass = "MVTecAD"
            data_args = [
                "--data",
                dataclass,
                "--data.root",
                data_root,
                "--data.category",
                "dummy",
            ]
        else:
            data_args = []

        return [
            "--model",
            "Padim",
            *data_args,
            "--default_root_dir",
            str(project_path),
            "--trainer.max_epochs",
            "1",
        ]
