"""Fixtures for the sweep tests."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from tempfile import TemporaryDirectory

import pytest

from anomalib.data import MVTec
from anomalib.deploy import ExportMode
from anomalib.engine import Engine
from anomalib.models import Padim
from anomalib.utils.types import TaskType


@pytest.fixture(scope="package")
def generate_results_dir():
    with TemporaryDirectory() as project_path:

        def make(
            path: str,
            category: str = "shapes",
            export_mode: ExportMode = ExportMode.OPENVINO,
        ) -> str:
            model = Padim()
            datamodule = MVTec(root=path, category=category)
            engine = Engine(
                logger=False,
                default_root_dir=project_path,
                task=TaskType.CLASSIFICATION,
                fast_dev_run=True,
                max_epochs=1,
                devices=1,
            )
            engine.fit(model=model, datamodule=datamodule)
            engine.export(model=model, task=TaskType.CLASSIFICATION, datamodule=datamodule, export_mode=export_mode)

            return project_path

        yield make
