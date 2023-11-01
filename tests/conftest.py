"""Fixtures for the entire test suite."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Generator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from anomalib.data import ImageDataFormat, TaskType, VideoDataFormat
from anomalib.data.base import dataset
from tests.helpers.data import DummyImageDatasetGenerator, DummyVideoDatasetGenerator
from tests.legacy.helpers.dataset import GeneratedDummyDataset


def _model_names() -> list[str]:
    """Return list of strings so that pytest can show the entire path of the yaml in the test log.

    Path object is not serializable by pytest.
    """
    # TODO(ashwinvaidya17)
    # Restore testing of ai_vad model.
    return [str(path.stem) for path in Path("src/configs/model").glob("*.yaml") if path.stem != "ai_vad"]


def _dataset_names() -> list[str]:
    # return [str(path.stem) for path in Path("src/configs/data").glob("*.yaml")]
    return ["mvtec"]


@pytest.fixture(scope="session")
def project_path() -> Generator[str, Any, None]:
    """Fixture to create a temporary directory for the project.

    Yields:
        Generator[str, Any, None]: Temporary directory path.
    """
    with TemporaryDirectory() as project_path:
        yield project_path


@pytest.fixture(scope="session")
def dataset_path(project_path: str) -> str:
    """Fixture that returns the path to the dummy datasets.

    Args:
        project_path (str): Project path that is created in /tmp

    Returns:
        str: Path to the dummy datasets
    """
    return f"{project_path}/datasets"


@pytest.fixture(scope="session")
def dataset_root(project_path: str, dataset_name: str) -> Generator[Path, Any, None]:
    """Fixture to create a dataset root.

    Args:
        project_path (str): Path to the project.
        dataset_name (str): Name of the dataset to create and return.

    Raises:
        ValueError: When dataset name is not supported.

    Yields:
        Generator[Path, Any, None]: Path to the dataset root.
    """
    if dataset_name in list(ImageDataFormat):
        with DummyImageDatasetGenerator(data_format=dataset_name, root=project_path) as data_root:
            yield data_root
    elif dataset_name in list(VideoDataFormat):
        with DummyVideoDatasetGenerator(data_format=dataset_name, root=project_path) as data_root:
            yield data_root
    else:
        message = (
            f"Dataset {dataset_name} is not supported. "
            "Please choose from {list(ImageDataFormat)} or {list(VideoDataFormat)}."
        )
        raise ValueError(message)


@pytest.fixture(params=[TaskType.CLASSIFICATION, TaskType.DETECTION, TaskType.SEGMENTATION])
def task_type(request: type[pytest.FixtureRequest]) -> str:
    """Create and return a task type."""
    return request.param


@pytest.fixture(scope="session", params=_model_names())
def model_name(request):
    return request.param


@pytest.fixture(scope="session", params=_dataset_names())
def dataset_name(request: type[pytest.FixtureRequest]) -> str:
    """Return a dataset name from the available datasets."""
    return request.param
