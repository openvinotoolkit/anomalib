"""Unit Tests - Avenue Datamodule."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from anomalib.data import Avenue, TaskType
from tests.helpers.data import get_dataset_path

from .base import _TestAnomalibVideoDatamodule


class TestAvenue(_TestAnomalibVideoDatamodule):
    """Avenue Datamodule Unit Tests."""

    @pytest.fixture
    def datamodule(self, task_type: TaskType) -> Avenue:
        # Create and prepare the dataset
        root = get_dataset_path("avenue")
        gt_dir = os.path.join(root, "ground_truth_demo")

        # TODO: Fix the Avenue dataset path via get_dataset_path
        _datamodule = Avenue(
            root=root,
            gt_dir=gt_dir,
            image_size=256,
            task=task_type,
            num_workers=0,
            train_batch_size=4,
            eval_batch_size=4,
        )

        _datamodule.prepare_data()
        _datamodule.setup()

        return _datamodule
