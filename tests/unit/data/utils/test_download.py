# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for download utils."""

import io
import tarfile
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from anomalib.data.utils.download import extract, is_within_directory


class TestIsWithinDirectory:
    """Tests for ``is_within_directory`` function."""

    @staticmethod
    def test_path_inside_directory() -> None:
        """Test that a path inside the directory is correctly identified."""
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            target = root / "safe" / "file.txt"
            assert is_within_directory(root, target) is True

    @staticmethod
    def test_path_traversal_outside_directory() -> None:
        """Test that a path escaping the directory via ``..`` is rejected."""
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "root"
            target = root / ".." / ".." / "etc" / "passwd"
            assert is_within_directory(root, target) is False

    @staticmethod
    def test_absolute_path_outside_directory() -> None:
        """Test that an absolute path outside the directory is rejected."""
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "root"
            target = Path("/etc/passwd")
            assert is_within_directory(root, target) is False


class TestExtract:
    """Tests for ``extract`` function against malicious archives."""

    @staticmethod
    def test_tar_path_traversal_is_blocked() -> None:
        """Test that tar members with ``../`` traversal are not extracted outside root."""
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            archive_path = base / "malicious.tar.gz"
            extract_root = base / "extracted"
            extract_root.mkdir()
            outside_marker = base / "evil.txt"

            with tarfile.open(archive_path, "w:gz") as tar:
                # Safe member should still be extracted normally.
                safe_info = tarfile.TarInfo(name="safe.txt")
                safe_data = b"safe content"
                safe_info.size = len(safe_data)
                tar.addfile(safe_info, fileobj=io.BytesIO(safe_data))

                # Traversal member should be rejected.
                evil_info = tarfile.TarInfo(name="../evil.txt")
                evil_data = b"evil content"
                evil_info.size = len(evil_data)
                tar.addfile(evil_info, fileobj=io.BytesIO(evil_data))

            extract(archive_path, extract_root)

            assert (extract_root / "safe.txt").exists()
            assert not outside_marker.exists()

    @staticmethod
    def test_tar_absolute_path_is_blocked() -> None:
        """Test that tar members with absolute paths are not extracted outside root."""
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            archive_path = base / "malicious_abs.tar.gz"
            extract_root = base / "extracted"
            extract_root.mkdir()
            outside_marker = base / "abs_evil.txt"

            with tarfile.open(archive_path, "w:gz") as tar:
                evil_info = tarfile.TarInfo(name=str(outside_marker))
                evil_data = b"evil content"
                evil_info.size = len(evil_data)
                tar.addfile(evil_info, fileobj=io.BytesIO(evil_data))

            extract(archive_path, extract_root)

            assert not outside_marker.exists()

    @staticmethod
    def test_tar_symlink_member_is_blocked() -> None:
        """Test that non-regular tar members (e.g. symlinks) are not extracted."""
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            archive_path = base / "malicious_symlink.tar.gz"
            extract_root = base / "extracted"
            extract_root.mkdir()

            with tarfile.open(archive_path, "w:gz") as tar:
                safe_info = tarfile.TarInfo(name="safe.txt")
                safe_data = b"safe content"
                safe_info.size = len(safe_data)
                tar.addfile(safe_info, fileobj=io.BytesIO(safe_data))

                link_info = tarfile.TarInfo(name="evil_link")
                link_info.type = tarfile.SYMTYPE
                link_info.linkname = "/etc/passwd"
                tar.addfile(link_info)

            extract(archive_path, extract_root)

            assert (extract_root / "safe.txt").exists()
            assert not (extract_root / "evil_link").exists()

    @staticmethod
    def test_zip_path_traversal_is_blocked() -> None:
        """Test that zip members with ``../`` traversal are not extracted outside root."""
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir)
            archive_path = base / "malicious.zip"
            extract_root = base / "extracted"
            extract_root.mkdir()
            outside_marker = base / "evil.txt"

            with zipfile.ZipFile(archive_path, "w") as zip_file:
                zip_file.writestr("safe.txt", "safe content")
                zip_file.writestr("../evil.txt", "evil content")

            extract(archive_path, extract_root)

            assert (extract_root / "safe.txt").exists()
            assert not outside_marker.exists()
