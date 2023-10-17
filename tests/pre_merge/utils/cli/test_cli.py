"""Test CLI entrypoints."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from anomalib.utils.cli import AnomalibCLI
from tests.helpers.dataset import GeneratedDummyDataset


def get_model_configs() -> list[str]:
    """Return list of strings so that pytest can show the entire path of the yaml in the test log.

    Path object is not serializable by pytest.
    """
    return [str(path) for path in Path("src/configs/model").glob("*.yaml")]


class TestCLI:
    """Do sanity check on all models."""

    @pytest.fixture(scope="class")
    def dataset(self):
        """Generate a dummy dataset."""
        with GeneratedDummyDataset(num_train=20, num_test=10) as dataset_path:
            yield dataset_path

    @pytest.fixture(scope="class")
    def project_path(self):
        with TemporaryDirectory() as project_path:
            yield project_path

    @pytest.mark.parametrize("model_config", get_model_configs())
    def test_fit(self, model_config: str, dataset: str, project_path: str):
        AnomalibCLI(
            args=[
                "fit",
                "-c",
                str(model_config),
                "--data",
                "anomalib.data.MVTec",
                "--data.root",
                dataset,
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

    @pytest.mark.parametrize("model_config", get_model_configs())
    def test_test(self, model_config: str, dataset: str, project_path: str):
        AnomalibCLI(
            args=[
                "test",
                "-c",
                str(model_config),
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
                f"{project_path}/{model_config.rstrip('.yaml')}/mvtec/shapes/weights/lightning/model.ckpt",
            ]
        )

    # TODO Validate
    # @pytest.mark.parametrize("model_config", get_model_configs())
    # def test_validate(self, model_config:str, dataset:str, project_path:str):
    #     AnomalibCLI(
    #         args=[
    #             "validate",
    #             "-c",
    #             str(model_config),
    #             "--data",
    #             "anomalib.data.MVTec",
    #             "--data.root",
    #             dataset,
    #             "--data.category",
    #             "shapes",
    #             "--results_dir.path",
    #             project_path,
    #             "--results_dir.unique",
    #             "false",
    #             "--ckpt_path",
    #             f"{project_path}/{model_config.rstrip(".yaml")}/mvtec/shapes/weights/lightning/model.ckpt",
    #         ]
    #     )

    # TODO Predict

    # TODO export
