"""Fixtures for the entire test suite."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Generator
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from anomalib.data import ImageDataFormat, VideoDataFormat
from tests.helpers.data import DummyImageDatasetGenerator, DummyVideoDatasetGenerator


def _dataset_names() -> list[str]:
    return [str(path.stem) for path in Path("src/configs/data").glob("*.yaml")]


@pytest.fixture(scope="session")
def project_path() -> Generator[Path, None, None]:
    """Return a temporary directory path that is used as the project directory for the entire test."""
    with TemporaryDirectory() as project_path:
        yield Path(project_path)


@pytest.fixture(scope="session")
def dataset_path(project_path: Path) -> Path:
    """Return a temporary directory path that is used as the dataset directory for the entire test.

    This fixture first generates the dummy datasets and return the dataset path before any other tests are run.
    Overall, the fixture does the following:

    1. Generate the image dataset.
    2. Generate the video dataset.
    3. Return the dataset path that contains the dummy datasets.
    """
    _dataset_path = project_path / "datasets"

    # 1. Create the dummy image datasets.
    for data_format in list(ImageDataFormat):
        # Do not generate a dummy dataset for folder datasets.
        # We could use one of these datasets to test the folders datasets.
        if not data_format.value.startswith("folder"):
            dataset_generator = DummyImageDatasetGenerator(data_format=data_format, root=_dataset_path)
            dataset_generator.generate_dataset()

    # 2. Create the dummy video datasets.
    for data_format in list(VideoDataFormat):
        dataset_generator = DummyVideoDatasetGenerator(data_format=data_format, root=_dataset_path)
        dataset_generator.generate_dataset()

    # 3. Return the dataset path.
    return _dataset_path


@pytest.fixture(scope="session", params=_dataset_names())
def dataset_name(request: "pytest.FixtureRequest") -> list[str]:
    """Return the list of names of all the datasets."""
    return request.param
