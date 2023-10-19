"""Test CLI entrypoints."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from anomalib.utils.cli import AnomalibCLI
from tests.helpers.dataset import get_dataset_path


class TestCLI:
    """Do sanity check on all models."""

    def test_fit(self, model_name: str, dataset_root: str, project_path: str):
        data_class = "anomalib.data.MVTec"
        category = "shapes"
        if model_name == "ai_vad":
            # TODO(ashwinva)
            # fix this when the dummy dataset supports all the data formats
            # aivad expects UCSD dataset
            data_class = "anomalib.data.UCSDped"
            dataset_root = get_dataset_path(dataset="ucsd")
            category = "UCSDped2"

        AnomalibCLI(
            args=[
                "fit",
                "-c",
                f"src/configs/model/{model_name}.yaml",
                "--data",
                data_class,
                "--data.root",
                dataset_root,
                "--data.category",
                category,
                "--results_dir.path",
                project_path,
                "--results_dir.unique",
                "false",
            ]
        )

    def test_test(self, model_name: str, dataset_root: str, project_path: str):
        data_class = "anomalib.data.MVTec"
        category = "shapes"
        if model_name == "ai_vad":
            # TODO(ashwinva)
            # fix this when the dummy dataset supports all the data formats
            # aivad expects UCSD dataset
            data_class = "anomalib.data.UCSDped"
            dataset_root = get_dataset_path(dataset="ucsd")
            category = "UCSDped2"

        AnomalibCLI(
            args=[
                "test",
                "-c",
                f"src/configs/model/{model_name}.yaml",
                "--data",
                data_class,
                "--data.root",
                dataset_root,
                "--data.category",
                category,
                "--results_dir.path",
                project_path,
                "--results_dir.unique",
                "false",
                "--ckpt_path",
                f"{project_path}/{model_name}/mvtec/shapes/weights/lightning/model.ckpt",
            ]
        )

    def test_validate(self, model_name: str, dataset_root: str, project_path: str):
        data_class = "anomalib.data.MVTec"
        category = "shapes"
        if model_name == "ai_vad":
            # TODO(ashwinva)
            # fix this when the dummy dataset supports all the data formats
            # aivad expects UCSD dataset
            data_class = "anomalib.data.UCSDped"
            dataset_root = get_dataset_path(dataset="ucsd")
            category = "UCSDped2"

        AnomalibCLI(
            args=[
                "validate",
                "-c",
                f"src/configs/model/{model_name}.yaml",
                "--data",
                data_class,
                "--data.root",
                dataset_root,
                "--data.category",
                category,
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
