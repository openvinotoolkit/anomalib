"""Fixtures for anomalib test suite."""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tests.helpers.dataset import DummyDatasetGenerator

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
def project_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Fixture that returns a temporary project path."""
    # TODO (samet-akcay): Check if this could be Path object instead of str.
    return str(tmp_path_factory.mktemp("project"))


@pytest.fixture(scope="session")
def dataset_root() -> Generator[str, Any, None]:
    """Generate a dummy dataset."""
    # TODO (samet-akcay): Pass ``data_format`` as a parameter to the fixture.
    # TODO (samet-akcay): Check if this could be Path object instead of str.
    with DummyDatasetGenerator(data_format="mvtec", num_train=10, num_test=5) as data_path:
        yield data_path


@pytest.fixture(scope="session", params=_model_names())
def model_name(request: type[pytest.FixtureRequest]) -> str:
    """Fixture that returns a model name."""
    return request.param


@pytest.fixture(scope="session", params=_dataset_names())
def dataset_name(request: type[pytest.FixtureRequest]) -> str:
    """Fixture that returns a dataset name."""
    return request.param
