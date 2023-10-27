"""Fixtures for anomalib test suite."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import pytest

from tests.legacy.helpers.dataset import GeneratedDummyDataset

if TYPE_CHECKING:
    from collections.abc import Generator


def _model_names() -> list[str]:
    """Return list of model names.

    Return as strings so that pytest can show the entire path of the yaml in the test log.
    Path object is not serializable by pytest.
    """
    # TODO(ashwinvaidya17)
    # Restore testing of ai_vad model.
    return [str(path.stem) for path in Path("src/configs/model").glob("*.yaml") if path.stem != "ai_vad"]


def _dataset_names() -> list[str]:
    return [str(path.stem) for path in Path("src/configs/data").glob("*.yaml")]


@pytest.fixture(scope="session")
def project_path() -> Generator[str, Any, None]:
    """Fixture that returns a temporary project path."""
    with TemporaryDirectory() as project_path:
        yield project_path


@pytest.fixture(scope="session")
def dataset_root() -> Generator[str, Any, None]:
    """Generate a dummy dataset."""
    with GeneratedDummyDataset(num_train=20, num_test=10) as data_root:
        yield data_root


@pytest.fixture(scope="session", params=_model_names())
def model_name(request: type[pytest.FixtureRequest]) -> str:
    """Fixture that returns a model name."""
    return request.param


@pytest.fixture(scope="session", params=_dataset_names())
def dataset_name(request: type[pytest.FixtureRequest]) -> str:
    """Fixture that returns a dataset name."""
    return request.param
