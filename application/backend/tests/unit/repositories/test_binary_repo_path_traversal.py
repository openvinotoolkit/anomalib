# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for path traversal in BinaryRepository.

Verifies that save_file, read_file, and delete_file refuse to operate on
filenames that resolve outside the project directory boundary.
"""

import asyncio
import os
from unittest.mock import patch

import pytest

from repositories.binary_repo import VideoBinaryRepository

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRAVERSAL_FILENAMES = [
    "../../../../tmp/pwned.mp4",
    "../sibling_project/video.mp4",
    "subdir/../../outside.mp4",
    # Absolute paths
    "/tmp/absolute.mp4",
    # Windows-style (only relevant when running on Windows)
    pytest.param(
        "..\\..\\outside.mp4",
        marks=pytest.mark.skipif(os.sep != "\\", reason="Backslash is not a path separator on POSIX"),
    ),
]

SAFE_FILENAME = "legitimate_video.mp4"


def run(coro):
    """Run a coroutine synchronously."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# BinaryRepository.save_file — path traversal must be rejected
# ---------------------------------------------------------------------------


class TestSaveFilePathTraversal:
    """save_file must raise ValueError for any filename that escapes the project dir."""

    @pytest.mark.parametrize("filename", TRAVERSAL_FILENAMES)
    def test_save_file_rejects_traversal(self, tmp_path, filename):
        """Traversal filename must raise ValueError, never write to disk."""
        project_dir = tmp_path / "projects" / "proj"
        project_dir.mkdir(parents=True)
        repo = VideoBinaryRepository(project_id="proj")

        with (
            patch.object(
                type(repo),
                "project_folder_path",
                new_callable=lambda: property(lambda self: str(project_dir)),
            ),
            pytest.raises((ValueError, OSError)),
        ):
            run(repo.save_file(filename=filename, content=b"PWNED"))

        # No additional filesystem assertions needed here: if traversal is not blocked,
        # repo.save_file would succeed and the pytest.raises(...) above would fail.

    def test_save_file_accepts_safe_filename(self, tmp_path):
        """A clean filename must be written inside the project directory."""
        project_dir = tmp_path / "projects" / "proj"
        project_dir.mkdir(parents=True)
        repo = VideoBinaryRepository(project_id="52138d75-cc61-4890-b8ba-ce6a619f0863")

        with patch.object(
            type(repo),
            "project_folder_path",
            new_callable=lambda: property(lambda self: str(project_dir)),
        ):
            saved = run(repo.save_file(filename=SAFE_FILENAME, content=b"OK"))

        assert os.path.isfile(saved)
        assert saved.startswith(str(project_dir))

    def test_save_file_bad_payload(self, tmp_path):
        """Malformed payload must be blocked."""
        project_dir = tmp_path / "data" / "videos" / "projects" / "52138d75-cc61-4890-b8ba-ce6a619f0863"
        project_dir.mkdir(parents=True)
        repo = VideoBinaryRepository(project_id="52138d75-cc61-4890-b8ba-ce6a619f0863")

        traversal = "../../anomalib_pwned.mp4"
        with (
            patch.object(
                type(repo),
                "project_folder_path",
                new_callable=lambda: property(lambda self: str(project_dir)),
            ),
            pytest.raises((ValueError, OSError)),
        ):
            run(repo.save_file(filename=traversal, content=b"PWNED"))

        assert not (tmp_path / "data" / "videos" / "anomalib_pwned.mp4").exists()


# ---------------------------------------------------------------------------
# BinaryRepository.read_file — path traversal must be rejected
# ---------------------------------------------------------------------------


class TestReadFilePathTraversal:
    """read_file must not serve files outside the project dir."""

    def test_read_file_rejects_traversal(self, tmp_path):
        """Traversal filename must raise ValueError and must not serve files outside the project dir."""
        project_dir = tmp_path / "projects" / "proj"
        project_dir.mkdir(parents=True)
        (tmp_path / "secret.mp4").write_bytes(b"SECRET")

        repo = VideoBinaryRepository(project_id="proj")
        traversal = os.path.join("..", "..", "secret.mp4")
        with (
            patch.object(
                type(repo),
                "project_folder_path",
                new_callable=lambda: property(lambda self: str(project_dir)),
            ),
            pytest.raises(ValueError),
        ):
            run(repo.read_file(filename=traversal))


# ---------------------------------------------------------------------------
# BinaryRepository.delete_file — path traversal must be rejected
# ---------------------------------------------------------------------------


class TestDeleteFilePathTraversal:
    """delete_file must not remove files outside the project dir."""

    def test_delete_file_rejects_traversal(self, tmp_path):
        """Traversal filename must raise ValueError; target file must survive."""
        project_dir = tmp_path / "projects" / "proj"
        project_dir.mkdir(parents=True)
        target = tmp_path / "important.mp4"
        target.write_bytes(b"KEEP ME")

        repo = VideoBinaryRepository(project_id="proj")
        traversal = os.path.join("..", "..", "important.mp4")
        with (
            patch.object(
                type(repo),
                "project_folder_path",
                new_callable=lambda: property(lambda self: str(project_dir)),
            ),
            pytest.raises(ValueError),
        ):
            run(repo.delete_file(filename=traversal))

        assert target.exists()
