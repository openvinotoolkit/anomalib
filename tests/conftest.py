"""Fixtures for the entire test suite."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable, Generator
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from lightning.pytorch.callbacks import ModelCheckpoint

from anomalib.data import ImageDataFormat, MVTec, VideoDataFormat
from anomalib.engine import Engine
from anomalib.models import get_model
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


@pytest.fixture(scope="session")
def ckpt_path(project_path: Path, dataset_path: Path) -> Callable[[str], Path]:
    """Return the checkpoint path of the trained model."""

    def checkpoint(model_name: str) -> Path:
        """Return the path to the trained model.

        Since integration tests train all the models, model training occurs when running unit tests invididually.
        """
        _ckpt_path = project_path / model_name.lower() / "dummy" / "weights" / "last.ckpt"
        if not _ckpt_path.exists():
            model = get_model(model_name)
            engine = Engine(
                logger=False,
                default_root_dir=project_path,
                max_epochs=1,
                devices=1,
                callbacks=[
                    ModelCheckpoint(
                        dirpath=project_path / model_name.lower() / "dummy" / "weights",
                        monitor=None,
                        filename="last",
                        save_last=True,
                        auto_insert_metric_name=False,
                    ),
                ],
            )
            dataset = MVTec(root=dataset_path / "mvtec", category="dummy")
            engine.fit(model=model, datamodule=dataset)

        return _ckpt_path

    return checkpoint