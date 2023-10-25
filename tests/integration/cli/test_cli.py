"""Test CLI entrypoints."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import torch
from tests.legacy.helpers.dataset import get_dataset_path

from anomalib.utils.cli import AnomalibCLI


class TestCLI:
    """Do sanity check on all models."""

    def test_fit(self, model_name: str, dataset_root: str, project_path: str):
        """Test fit CLI.

        Args:
            model_name (str): Name of the model to train.
            dataset_root (str): Root of the synthetic/original dataset.
            project_path (str): Path to temporary project folder.
        """
        data_class = "anomalib.data.MVTec"
        category = "shapes"
        if model_name == "ai_vad":
            # TODO(ashwinva)
            # fix this when the dummy dataset supports all the data formats
            # aivad expects UCSD dataset
            data_class = "anomalib.data.UCSDped"
            dataset_root = get_dataset_path(dataset="ucsd")
            category = "UCSDped2"

        # batch size of 8 is taken so that the lr computation for efficient_ad does not return 0 when max_epochs is 1
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
                "--data.train_batch_size",
                "8",
                "--results_dir.path",
                project_path,
                "--results_dir.unique",
                "false",
                "--trainer.max_epochs",
                "1",
                "--trainer.check_val_every_n_epoch",
                "1",
                "--trainer.callbacks+=ModelCheckpoint",
                "--trainer.callbacks.dirpath",
                f"{project_path}/{model_name}/{data_class.split('.')[-1].lower()}/{category}/weights",
                "--trainer.callbacks.monitor",
                "null",
                "--trainer.callbacks.filename",
                "last",
                "--trainer.callbacks.save_last",
                "true",
                "--trainer.callbacks.auto_insert_metric_name",
                "false",
            ]
        )
        torch.cuda.empty_cache()

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
                f"{project_path}/{model_name}/{data_class.split('.')[-1].lower()}/{category}/weights/last.ckpt",
            ]
        )
        torch.cuda.empty_cache()

    # def test_validate(self, model_name: str, dataset_root: str, project_path: str):
    #     data_class = "anomalib.data.MVTec"
    #     category = "shapes"
    #     if model_name == "ai_vad":
    #         # TODO(ashwinva)
    #         # fix this when the dummy dataset supports all the data formats
    #         # aivad expects UCSD dataset
    #         data_class = "anomalib.data.UCSDped"
    #         dataset_root = get_dataset_path(dataset="ucsd")
    #         category = "UCSDped2"

    #     AnomalibCLI(
    #         args=[
    #             "validate",
    #             "-c",
    #             f"src/configs/model/{model_name}.yaml",
    #             "--data",
    #             data_class,
    #             "--data.root",
    #             dataset_root,
    #             "--data.category",
    #             category,
    #             "--results_dir.path",
    #             project_path,
    #             "--results_dir.unique",
    #             "false",
    #             "--ckpt_path",
    #             f"{project_path}/{model_name}/{data_class.split('.')[-1].lower()}/{category}/weights/last.ckpt",
    #         ]
    #     )
    #     torch.cuda.empty_cache()

    # TODO(ashwinva) Predict

    # TODO(ashwinva) export
