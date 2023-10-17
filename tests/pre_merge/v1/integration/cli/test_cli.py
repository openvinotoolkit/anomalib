"""Test CLI entrypoints."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from anomalib.utils.cli import AnomalibCLI


class TestCLI:
    """Do sanity check on all models."""

    def test_fit(self, model_name: str, dataset_root: str, project_path: str):
        AnomalibCLI(
            args=[
                "fit",
                "-c",
                f"src/configs/model/{model_name}.yaml",
                "--data",
                "anomalib.data.MVTec",
                "--data.root",
                dataset_root,
                "--data.category",
                "shapes",
                "--trainer.max_epochs",
                "1",
                "--results_dir.path",
                project_path,
                "--results_dir.unique",
                "false",
            ]
        )

    def test_test(self, model_name: str, dataset_root: str, project_path: str):
        AnomalibCLI(
            args=[
                "test",
                "-c",
                f"src/configs/model/{model_name}.yaml",
                "--data",
                "anomalib.data.MVTec",
                "--data.root",
                dataset_root,
                "--data.category",
                "shapes",
                "--results_dir.path",
                project_path,
                "--results_dir.unique",
                "false",
                "--ckpt_path",
                f"{project_path}/{model_name}/mvtec/shapes/weights/lightning/model.ckpt",
            ]
        )

    def test_validate(self, model_name: str, dataset: str, project_path: str):
        AnomalibCLI(
            args=[
                "validate",
                "-c",
                f"src/configs/model/{model_name}.yaml",
                "--data",
                "anomalib.data.MVTec",
                "--data.root",
                dataset,
                "--data.category",
                "shapes",
                "--results_dir.path",
                project_path,
                "--results_dir.unique",
                "false",
                "--ckpt_path",
                f"{project_path}/{model_name}/mvtec/shapes/weights/lightning/model.ckpt",
            ]
        )

    # TODO(ashwinva) Predict

    # TODO(ashwinva) export
