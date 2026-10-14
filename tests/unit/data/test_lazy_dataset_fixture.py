# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Regression: dummy datasets generate lazily per format."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from anomalib.data import ImageDataFormat
from tests.fixtures.datasets import build_make_dummy_dataset

_LAZY_FIXTURE_CHILD = "ANOMALIB_LAZY_FIXTURE_CHILD"


def test_make_dummy_dataset_creates_only_requested_format(tmp_path: Path) -> None:
    """Only the requested format directory is created on first use."""
    make_dummy_dataset = build_make_dummy_dataset(tmp_path)
    excluded_formats = {"folder", "tabular", "folder_3d"}

    assert tmp_path.is_dir()
    for fmt in ImageDataFormat:
        if fmt.value not in excluded_formats:
            assert not (tmp_path / fmt.value).exists()

    root = make_dummy_dataset("mvtecad")
    assert root == tmp_path
    assert (tmp_path / "mvtecad").is_dir()
    # Other formats must not appear as a side effect of MVTecAD generation.
    for fmt in ImageDataFormat:
        if fmt.value == "mvtecad" or fmt.value in excluded_formats:
            continue
        assert not (tmp_path / fmt.value).exists()


@pytest.mark.skipif(os.environ.get(_LAZY_FIXTURE_CHILD) != "1", reason="run only in isolated subprocess")
def test_dataset_path_is_empty_on_first_request(dataset_path: Path) -> None:
    """The session fixture itself must not eagerly generate dataset formats."""
    assert not any((dataset_path / fmt.value).exists() for fmt in ImageDataFormat)


def test_session_dataset_path_fixture_is_lazy() -> None:
    """Request ``dataset_path`` in an isolated pytest session."""
    repo_root = Path(__file__).parents[3]
    env = os.environ.copy()
    env[_LAZY_FIXTURE_CHILD] = "1"
    result = subprocess.run(  # noqa: S603 - arguments are fixed by this test.
        [
            sys.executable,
            "-m",
            "pytest",
            f"{__file__}::test_dataset_path_is_empty_on_first_request",
            "-q",
            "--tb=line",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
