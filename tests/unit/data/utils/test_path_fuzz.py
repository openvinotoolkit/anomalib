# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Bounded fuzzing of the path boundary used by untrusted dataset metadata."""

import os
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from anomalib.data.utils.path import resolve_path_under_root

# Keep ordinary unit runs bounded; CI records its seed for exact replay.
FUZZ_SETTINGS = settings(max_examples=1000, deadline=None, print_blob=True)
SAFE_PARTS = st.lists(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_", min_size=1, max_size=16).map(lambda part: f"dir_{part}"),
    max_size=6,
)
METADATA_PATHS = st.one_of(
    st.text(max_size=600),
    st.lists(
        st.sampled_from(["..", ".", "", "images", "image.png", "C:", "\\", "\x00", "%2e%2e", "outside"]),
        min_size=1,
        max_size=20,
    ).map("/".join),
)


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a stable root; fuzz inputs are resolved but never written."""
    root = tmp_path_factory.mktemp("metadata-fuzz") / "dataset"
    root.mkdir()
    return root.resolve()


@FUZZ_SETTINGS
@given(value=METADATA_PATHS, anchor=st.sampled_from(["relative", "inside", "outside"]))
@example(value="../outside.png", anchor="relative")
@example(value="../../dataset-sibling/image.png", anchor="relative")
@example(value="/etc/passwd", anchor="relative")
@example(value="C:\\Windows\\win.ini", anchor="relative")
@example(value="\\\\server\\share\\image.png", anchor="relative")
@example(value="image\x00.png", anchor="relative")
@example(value="image\n.png", anchor="relative")
@example(value="a" * 513, anchor="relative")
def test_metadata_path_fuzz(dataset_root: Path, value: str, anchor: str) -> None:
    """Every accepted metadata path must stay within the dataset root."""
    if os.name == "nt" and value.replace("/", "\\").startswith("\\\\"):
        # shortcut: UNC paths can contact remote hosts; cover them in an isolated Windows runner.
        return
    if anchor == "inside":
        value = str(dataset_root / value)
    elif anchor == "outside":
        value = str(dataset_root.parent / value)
    for extensions in (None, (".png",)):
        try:
            result = resolve_path_under_root(dataset_root, value, should_exist=False, extensions=extensions)
        except (ValueError, OSError):
            # Validation errors and platform-invalid paths are legitimate rejections.
            continue

        # Use a separate containment oracle, not the production helper.
        assert os.path.commonpath([dataset_root, result]) == str(dataset_root)
        assert result.is_absolute()
        if extensions is not None:
            assert result.suffix in extensions


@FUZZ_SETTINGS
@given(parts=SAFE_PARTS)
@example(parts=[])
@example(parts=["images", "normal"])
def test_valid_metadata_path_fuzz(dataset_root: Path, parts: list[str]) -> None:
    """Valid paths must be accepted, preventing an always-rejecting oracle."""
    relative = Path(*parts) / "image.png"
    expected = dataset_root / relative
    assert resolve_path_under_root(dataset_root, relative, should_exist=False) == expected
    assert resolve_path_under_root(dataset_root, expected, should_exist=False) == expected
    with pytest.raises(ValueError, match="outside the allowed directory"):
        resolve_path_under_root(dataset_root, Path("..") / relative, should_exist=False)
    with pytest.raises(ValueError, match="outside the allowed directory"):
        resolve_path_under_root(dataset_root, dataset_root.parent / "dataset-sibling" / relative, should_exist=False)


@pytest.fixture(scope="module")
def escape_link(dataset_root: Path) -> Path:
    """Create a symlink crossing the root boundary when the OS permits it."""
    outside = dataset_root.parent / "outside"
    outside.mkdir()
    link = dataset_root / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        if os.name == "nt" and error.winerror == 1314:
            pytest.skip("Windows requires Developer Mode or symlink privileges")
        raise
    return link


@FUZZ_SETTINGS
@given(parts=SAFE_PARTS)
@example(parts=[])
def test_symlink_escape_fuzz(dataset_root: Path, escape_link: Path, parts: list[str]) -> None:
    """Paths through an escape symlink must be rejected before file access."""
    candidate = escape_link.joinpath(*parts, "image.png")
    with pytest.raises(ValueError, match="outside the allowed directory"):
        resolve_path_under_root(dataset_root, candidate, should_exist=False)
