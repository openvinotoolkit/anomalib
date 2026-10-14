# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the entire test suite."""

from collections.abc import Generator
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

pytest_plugins = [
    "tests.fixtures.datasets",
    "tests.fixtures.models",
]


def _dataset_names() -> list[str]:
    return [str(path.stem) for path in Path("examples/configs/data").glob("*.yaml")]


@pytest.fixture(scope="session")
def project_path(pytestconfig: pytest.Config) -> Generator[Path, None, None]:
    """Return a temporary directory path that is used as the project directory for the entire test.

    Each pytest-xdist worker gets its own subdirectory under ``tmp/``. Only that
    subdirectory is cleaned up so parallel workers do not delete each other's trees.
    """
    # Get the root directory of the project
    root_dir = Path(__file__).parent.parent

    # Create the temporary directory in the root directory of the project.
    # This is to access the test files in the project directory.
    # Only remove this session's TemporaryDirectory on exit — never wipe ``tmp/``
    # wholesale. Nested pytest sessions (e.g. lazy-fixture meta-tests) also create
    # siblings under ``tmp/``; deleting the parent would erase their (and our) data.
    tmp_dir = root_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    workerinput = getattr(pytestconfig, "workerinput", None)
    worker_id = workerinput["workerid"] if workerinput else "master"

    with TemporaryDirectory(prefix=f"anomalib-{worker_id}-", dir=tmp_dir) as tmp_sub_dir:
        project_path = Path(tmp_sub_dir)
        # Restrict permissions (read and write for owner only)
        project_path.chmod(0o700)
        yield project_path


@pytest.fixture(scope="session", autouse=True)
def _limit_torch_threads() -> None:
    """Limit PyTorch to one thread during tests.

    Avoids CPU oversubscription when pytest-xdist runs many workers, each of which
    would otherwise spawn many BLAS/OpenMP threads.
    """
    torch.set_num_threads(1)


@pytest.fixture(scope="session", params=_dataset_names())
def dataset_name(request: "pytest.FixtureRequest") -> list[str]:
    """Return the list of names of all the datasets."""
    return request.param


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Automatically mark tests as 'cpu' unless they're marked as 'gpu'."""
    for item in items:
        if not any(marker.name == "gpu" for marker in item.iter_markers()):
            item.add_marker(pytest.mark.cpu)
