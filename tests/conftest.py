"""Fixtures for the entire test suite."""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil
from collections.abc import Callable, Generator
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from anomalib.data import ImageDataFormat, MVTec, VideoDataFormat
from anomalib.engine import Engine
from anomalib.models import get_model
from tests.helpers.data import DummyImageDatasetGenerator, DummyVideoDatasetGenerator


def _dataset_names() -> list[str]:
    return [str(path.stem) for path in Path("configs/data").glob("*.yaml")]


@pytest.fixture(scope="session")
def project_path() -> Generator[Path, None, None]:
    """Return a temporary directory path that is used as the project directory for the entire test."""
    # Get the root directory of the project
    root_dir = Path(__file__).parent.parent

    # Create the temporary directory in the root directory of the project.
    # This is to access the test files in the project directory.
    tmp_dir = root_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    with TemporaryDirectory(dir=tmp_dir) as tmp_sub_dir:
        project_path = Path(tmp_sub_dir)
        # Restrict permissions (read and write for owner only)
        project_path.chmod(0o700)
        yield project_path

    # Clean up the temporary directory.
    shutil.rmtree(tmp_dir)


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


@pytest.fixture(scope="session")
def ckpt_path(project_path: Path, dataset_path: Path) -> Callable[[str], Path]:
    """Return the checkpoint path of the trained model."""

    def checkpoint(model_name: str) -> Path:
        """Return the path to the trained model.

        Since integration tests train all the models, model training occurs when running unit tests invididually.
        """
        model = get_model(model_name)
        _ckpt_path = project_path / model.name / "MVTec" / "dummy" / "latest" / "weights" / "lightning" / "model.ckpt"
        if not _ckpt_path.exists():
            engine = Engine(
                logger=False,
                default_root_dir=project_path,
                max_epochs=1,
                devices=1,
            )
            dataset = MVTec(root=dataset_path / "mvtec", category="dummy")
            engine.fit(model=model, datamodule=dataset)

        return _ckpt_path

    return checkpoint
