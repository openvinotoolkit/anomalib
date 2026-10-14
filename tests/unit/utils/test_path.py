# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for anomalib.utils.path."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from anomalib.utils.path import (
    _get_cache_subdir,
    create_versioned_dir,
    get_datasets_dir,
    get_pretrained_weights_dir,
    paths_to_strings,
    resolve_versioned_path,
)


class TestResolveVersionedPath:
    """Tests for resolve_versioned_path."""

    @staticmethod
    def test_path_without_latest_unchanged(tmp_path: Path) -> None:
        """Path with no 'latest' component is returned unchanged."""
        p = tmp_path / "some" / "path" / "file.ckpt"
        p.parent.mkdir(parents=True, exist_ok=True)
        assert resolve_versioned_path(p) == p
        assert resolve_versioned_path(str(p)) == p

    @staticmethod
    def test_path_with_latest_replaced_when_version_dirs_exist(tmp_path: Path) -> None:
        """When parent of 'latest' contains v0, v1, ..., 'latest' is replaced with highest vN."""
        root = tmp_path / "exp"
        (root / "v0").mkdir(parents=True)
        (root / "v1").mkdir()
        (root / "v2").mkdir()
        # On POSIX, resolve_versioned_path follows the symlink; create it so resolve() works
        if sys.platform != "win32":
            (root / "latest").symlink_to(root / "v2", target_is_directory=True)
        path_with_latest = root / "latest" / "weights" / "model.ckpt"
        (root / "v2" / "weights").mkdir(parents=True)
        resolved = resolve_versioned_path(path_with_latest)
        assert resolved == root / "v2" / "weights" / "model.ckpt"

    @staticmethod
    def test_path_with_latest_no_version_dirs_unchanged(tmp_path: Path) -> None:
        """When parent of 'latest' has no vN dirs, path is returned unchanged."""
        root = tmp_path / "exp"
        root.mkdir(parents=True)
        path_with_latest = root / "latest" / "weights" / "model.ckpt"
        resolved = resolve_versioned_path(path_with_latest)
        assert resolved == path_with_latest

    @staticmethod
    def test_path_with_latest_parent_missing_unchanged(tmp_path: Path) -> None:
        """When parent of 'latest' does not exist, path is returned unchanged."""
        path_with_latest = tmp_path / "nonexistent" / "latest" / "model.ckpt"
        resolved = resolve_versioned_path(path_with_latest)
        assert resolved == path_with_latest

    @staticmethod
    def test_path_with_latest_only_v0(tmp_path: Path) -> None:
        """When only v0 exists, 'latest' is replaced with v0."""
        root = tmp_path / "exp"
        (root / "v0").mkdir(parents=True)
        if sys.platform != "win32":
            (root / "latest").symlink_to(root / "v0", target_is_directory=True)
        path_with_latest = root / "latest" / "model.ckpt"
        resolved = resolve_versioned_path(path_with_latest)
        assert resolved == root / "v0" / "model.ckpt"

    @staticmethod
    def test_first_occurrence_of_latest_replaced(tmp_path: Path) -> None:
        """Only the first occurrence of 'latest' in path parts is replaced."""
        root = tmp_path / "exp"
        (root / "v1").mkdir(parents=True)
        if sys.platform != "win32":
            (root / "latest").symlink_to(root / "v1", target_is_directory=True)
        # path like .../latest/latest/file - first 'latest' resolved, second stays
        path_with_two = root / "latest" / "latest" / "file.ckpt"
        resolved = resolve_versioned_path(path_with_two)
        assert resolved == root / "v1" / "latest" / "file.ckpt"


class TestCreateVersionedDir:
    """Tests for create_versioned_dir (return value is real path)."""

    @staticmethod
    def test_returns_real_version_path_not_latest(tmp_path: Path) -> None:
        """create_versioned_dir returns the real version directory path, not the 'latest' link."""
        root = tmp_path / "results"
        result = create_versioned_dir(root)
        assert result.name == "v0"
        assert result == root / "v0"
        assert (root / "latest").exists()
        # Returned path must not be the link path
        assert result != root / "latest"

    @staticmethod
    def test_increments_version(tmp_path: Path) -> None:
        """Second call returns v1, etc."""
        root = tmp_path / "results"
        r1 = create_versioned_dir(root)
        r2 = create_versioned_dir(root)
        assert r1.name == "v0"
        assert r2.name == "v1"
        assert r1 != r2


class TestGetCacheSubdir:
    """Tests for _get_cache_subdir, get_pretrained_weights_dir, get_datasets_dir."""

    @staticmethod
    def _mock_cache(tmp_path: Path) -> Path:
        """Create and return a fake cache root (simulates ensure_exists=True)."""
        fake_cache = tmp_path / "cache"
        fake_cache.mkdir()
        return fake_cache

    def test_get_cache_subdir_returns_path(self, tmp_path: Path) -> None:
        """_get_cache_subdir returns a Path instance."""
        fake_cache = self._mock_cache(tmp_path)
        with patch("anomalib.utils.path.platformdirs.user_cache_path", return_value=fake_cache):
            result = _get_cache_subdir("some_dir")
        assert isinstance(result, Path)

    def test_get_cache_subdir_ends_with_subdir(self, tmp_path: Path) -> None:
        """Returned path ends with the requested subdirectory name."""
        fake_cache = self._mock_cache(tmp_path)
        with patch("anomalib.utils.path.platformdirs.user_cache_path", return_value=fake_cache):
            result = _get_cache_subdir("my_subdir")
        assert result.name == "my_subdir"

    def test_get_cache_subdir_creates_directory(self, tmp_path: Path) -> None:
        """The subdirectory is created on disk by the mkdir call."""
        fake_cache = self._mock_cache(tmp_path)
        subdir_path = fake_cache / "widgets"
        assert not subdir_path.exists()
        with patch("anomalib.utils.path.platformdirs.user_cache_path", return_value=fake_cache):
            result = _get_cache_subdir("widgets")
        assert result == subdir_path
        assert result.is_dir()

    def test_get_cache_subdir_idempotent(self, tmp_path: Path) -> None:
        """Calling twice with the same subdir does not raise and returns the same path."""
        fake_cache = self._mock_cache(tmp_path)
        with patch("anomalib.utils.path.platformdirs.user_cache_path", return_value=fake_cache):
            first = _get_cache_subdir("repeat")
            second = _get_cache_subdir("repeat")
        assert first == second
        assert first.is_dir()

    def test_get_pretrained_weights_dir_returns_path(self, tmp_path: Path) -> None:
        """get_pretrained_weights_dir returns a Path ending with 'pre_trained'."""
        fake_cache = self._mock_cache(tmp_path)
        with patch("anomalib.utils.path.platformdirs.user_cache_path", return_value=fake_cache):
            result = get_pretrained_weights_dir()
        assert isinstance(result, Path)
        assert result.name == "pre_trained"

    def test_get_pretrained_weights_dir_exists(self, tmp_path: Path) -> None:
        """Directory returned by get_pretrained_weights_dir exists on disk."""
        fake_cache = self._mock_cache(tmp_path)
        with patch("anomalib.utils.path.platformdirs.user_cache_path", return_value=fake_cache):
            result = get_pretrained_weights_dir()
        assert result.is_dir()

    def test_get_datasets_dir_returns_path(self, tmp_path: Path) -> None:
        """get_datasets_dir returns a Path ending with 'datasets'."""
        fake_cache = self._mock_cache(tmp_path)
        with patch("anomalib.utils.path.platformdirs.user_cache_path", return_value=fake_cache):
            result = get_datasets_dir()
        assert isinstance(result, Path)
        assert result.name == "datasets"

    def test_get_datasets_dir_exists(self, tmp_path: Path) -> None:
        """Directory returned by get_datasets_dir exists on disk."""
        fake_cache = self._mock_cache(tmp_path)
        with patch("anomalib.utils.path.platformdirs.user_cache_path", return_value=fake_cache):
            result = get_datasets_dir()
        assert result.is_dir()

    def test_ensure_exists_passed_to_platformdirs(self, tmp_path: Path) -> None:
        """platformdirs.user_cache_path is called with ensure_exists=True."""
        fake_cache = self._mock_cache(tmp_path)
        with patch("anomalib.utils.path.platformdirs.user_cache_path", return_value=fake_cache) as mock_ucp:
            _get_cache_subdir("test")
        mock_ucp.assert_called_once_with("anomalib", ensure_exists=True)

    @staticmethod
    def test_rejects_empty_string() -> None:
        """Empty string is rejected with ValueError."""
        with pytest.raises(ValueError, match="Invalid cache subdirectory name"):
            _get_cache_subdir("")

    @staticmethod
    def test_rejects_absolute_path() -> None:
        """Absolute path is rejected with ValueError."""
        with pytest.raises(ValueError, match="Invalid cache subdirectory name"):
            _get_cache_subdir("/etc/passwd")

    @staticmethod
    def test_rejects_dotdot_traversal() -> None:
        """Path containing '..' components is rejected with ValueError."""
        with pytest.raises(ValueError, match="Invalid cache subdirectory name"):
            _get_cache_subdir("../x")

    @staticmethod
    def test_rejects_dotdot_in_middle() -> None:
        """Path with '..' buried in the middle is rejected with ValueError."""
        with pytest.raises(ValueError, match="Invalid cache subdirectory name"):
            _get_cache_subdir("a/../b")

    @staticmethod
    def test_rejects_bare_dotdot() -> None:
        """Bare '..' is rejected with ValueError."""
        with pytest.raises(ValueError, match="Invalid cache subdirectory name"):
            _get_cache_subdir("..")


def test_paths_to_strings_converts_nested_paths() -> None:
    """Nested path values become strings without mutating input."""
    value = {"root": Path("data"), "items": [Path("one"), (Path("two"),)]}
    result = paths_to_strings(value)
    assert result == {"root": "data", "items": ["one", ("two",)]}
    assert value["root"] == Path("data")
