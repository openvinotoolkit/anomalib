"""Test CLI entrypoints.

This just checks if one of the model works end-to-end. The rest of the models are checked using the API.
"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import pytest
import torch

from anomalib.utils.cli import AnomalibCLI


class TestCLI:
    """Do sanity check on all models."""

    def test_fit(self, dataset_path: Path, project_path: Path) -> None:
        """Test fit CLI.

        Args:
            dataset_path (Path): Root of the synthetic/original dataset.
            project_path (Path): Path to temporary project folder.
        """
        # batch size of 8 is taken so that the lr computation for efficient_ad does not return 0 when max_epochs is 1
        AnomalibCLI(
            args=[
                "fit",
                "--model",
                "Padim",
                "--data",
                "MVTec",
                "--data.root",
                str(dataset_path / "mvtec"),
                "--data.category",
                "dummy",
                "--data.train_batch_size",
                "8",
                "--results_dir.path",
                str(project_path),
                "--results_dir.unique",
                "false",
                "--trainer.max_epochs",
                "1",
                "--trainer.check_val_every_n_epoch",
                "1",
                "--trainer.callbacks+=ModelCheckpoint",
                "--trainer.callbacks.dirpath",
                f"{project_path}/Padim/MVTec/dummy/weights",
                "--trainer.callbacks.monitor",
                "null",
                "--trainer.callbacks.filename",
                "last",
                "--trainer.callbacks.save_last",
                "true",
                "--trainer.callbacks.auto_insert_metric_name",
                "false",
            ],
        )
        torch.cuda.empty_cache()

    def test_test(self, dataset_path: Path, project_path: Path) -> None:
        """Test the test method of the CLI.

        Args:
            dataset_path (Path): Root of the synthetic/original dataset.
            project_path (Path): Path to temporary project folder.
        """
        AnomalibCLI(
            args=[
                "test",
                "--model",
                "Padim",
                "--data",
                "MVTec",
                "--data.root",
                str(dataset_path / "mvtec"),
                "--data.category",
                "dummy",
                "--results_dir.path",
                str(project_path),
                "--results_dir.unique",
                "false",
                "--ckpt_path",
                f"{project_path}/Padim/MVTec/dummy/weights/last.ckpt",
            ],
        )
        torch.cuda.empty_cache()

    @pytest.mark.skip(reason="validation is not implemented in Anomalib Engine")
    def test_validate(self, dataset_path: Path, project_path: Path) -> None:
        """Test the validate method of the CLI.

        Args:
            dataset_path (str): Root of the synthetic/original dataset.
            project_path (str): Path to temporary project folder.
        """
        AnomalibCLI(
            args=[
                "validate",
                "--model",
                "Padim",
                "--data",
                "MVTec",
                "--data.root",
                str(dataset_path / "mvtec"),
                "--data.category",
                "dummy",
                "--results_dir.path",
                str(project_path),
                "--results_dir.unique",
                "false",
                "--ckpt_path",
                f"{project_path}/Padim/MVTec/dummy/weights/last.ckpt",
            ],
        )
        torch.cuda.empty_cache()

    # TODO(ashwinvaidya17): Predict
    # CVS-109972

    # TODO(ashwinvaidya17): export
    # CVS-109972
