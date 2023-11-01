"""Unit Tests - MVTec Datamodule."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from anomalib.data import MVTec, TaskType

from .base.image import _TestAnomalibImageDatamodule

# class TestMVTec(_TestAnomalibImageDatamodule):
#     """MVTec Datamodule Unit Tests."""
#     # Create a dummy dataset generator using DummyImageDatasetGenerator
#     @pytest.fixture()
#     def dataset_name(self) -> str:
#         return "mvtec"

#     @pytest.fixture()
#     def datamodule(self, dataset_root, task_type: TaskType) -> MVTec:
#         # Create and prepare the dataset
#         _datamodule = MVTec(
#             root=dataset_root,
#             category="dummy",
#             task=task_type,
#             image_size=256,
#             train_batch_size=4,
#             eval_batch_size=4,
#         )
#         _datamodule.prepare_data()
#         _datamodule.setup()

#         return _datamodule
