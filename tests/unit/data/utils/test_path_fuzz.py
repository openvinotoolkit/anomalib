# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Replay the fuzz seed corpus and check that security violations fail."""

import os
from pathlib import Path

import pytest

from tests.fuzz.fuzz_path import fuzz_path, path_utils

CORPUS = Path(__file__).resolve().parents[3] / "fuzz" / "seeds" / "path"


@pytest.fixture
def dataset_root(tmp_path: Path) -> Path:
    """Create a resolved dataset directory for corpus replay."""
    root = tmp_path.resolve() / "dataset"
    root.mkdir()
    return root


@pytest.mark.parametrize("seed", sorted(CORPUS.iterdir()), ids=lambda seed: seed.name)
def test_path_corpus(dataset_root: Path, seed: Path) -> None:
    """Replay every checked-in seed without requiring Atheris."""
    data = seed.read_bytes()
    if os.name == "nt" and data.startswith(b"\\\\"):
        pytest.skip("UNC paths can contact remote hosts on Windows")
    fuzz_path(data, dataset_root)


def test_symlink_corpus(dataset_root: Path) -> None:
    """Require symlink escapes to fail on platforms with symlink privileges."""
    outside = dataset_root.parent / "outside"
    outside.mkdir()
    try:
        (dataset_root / "escape").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        if os.name == "nt" and error.winerror == 1314:
            pytest.skip("Windows requires Developer Mode or symlink privileges")
        raise
    for seed in sorted(CORPUS.iterdir()):
        if os.name != "nt" or not seed.read_bytes().startswith(b"\\\\"):
            fuzz_path(seed.read_bytes(), dataset_root)


def test_fuzzer_detects_escape(dataset_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove the oracle detects a containment check that accepts escapes."""
    monkeypatch.setattr(path_utils, "is_within_directory", lambda *_: True)
    with pytest.raises(AssertionError):
        fuzz_path(b"../outside.png", dataset_root)


def test_fuzzer_requires_valid_paths(dataset_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prove the oracle rejects an implementation that denies every path."""

    def reject(*_args: object, **_kwargs: object) -> Path:
        msg = "Rejected every path"
        raise ValueError(msg)

    monkeypatch.setattr(path_utils, "resolve_path_under_root", reject)
    with pytest.raises(ValueError, match="Rejected every path"):
        fuzz_path(b"image.png", dataset_root)
