# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for path utils."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from anomalib.data.utils.path import resolve_path_under_root, validate_path
from anomalib.utils.path import generate_output_filename


class TestValidatePath:
    """Tests for ``validate_path`` function."""

    @staticmethod
    def test_invalid_path_type() -> None:
        """Test ``validate_path`` raises TypeError for an invalid path type."""
        with pytest.raises(TypeError, match=r"Expected str, bytes or os.PathLike object, not*"):
            validate_path(123)

    @staticmethod
    def test_is_path_too_long() -> None:
        """Test ``validate_path`` raises ValueError for a path that is too long."""
        with pytest.raises(ValueError, match=r"Path is too long: *"):
            validate_path("/" * 1000)

    @staticmethod
    def test_contains_non_printable_characters() -> None:
        """Test ``validate_path`` raises ValueError for a path that contains non-printable characters."""
        with pytest.raises(ValueError, match=r"Path contains non-printable characters: *"):
            validate_path("/\x00")

    @staticmethod
    def test_existing_file_within_base_dir(mvtecad_path: Path) -> None:
        """Test ``validate_path`` returns the validated path for an existing file within the base directory."""
        file_path = mvtecad_path / "dummy/train/good/000.png"
        validated_path = validate_path(file_path, base_dir=mvtecad_path)
        assert validated_path == file_path.resolve()

    @staticmethod
    def test_existing_directory_within_base_dir(mvtecad_path: Path) -> None:
        """Test ``validate_path`` returns the validated path for an existing directory within the base directory."""
        directory_path = mvtecad_path / "dummy/train/good"
        validated_path = validate_path(directory_path, base_dir=mvtecad_path)
        assert validated_path == directory_path.resolve()

    @staticmethod
    def test_nonexistent_file(dataset_path: Path) -> None:
        """Test ``validate_path`` raises FileNotFoundError for a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            validate_path(dataset_path / "nonexistent/file.png")

    @staticmethod
    def test_nonexistent_directory(dataset_path: Path) -> None:
        """Test ``validate_path`` raises FileNotFoundError for a nonexistent directory."""
        with pytest.raises(FileNotFoundError):
            validate_path(dataset_path / "nonexistent/directory")

    @staticmethod
    def test_no_read_permission() -> None:
        """Test ``validate_path`` raises PermissionError for a file without read permission."""
        with TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test.txt"
            with file_path.open("w") as f:
                f.write("test")
            file_path.chmod(0o222)  # Remove read permission
            with pytest.raises(PermissionError, match=r"Read or execute permissions denied for the path:*"):
                validate_path(file_path, base_dir=Path(tmp_dir))

    @staticmethod
    def test_no_read_execute_permission() -> None:
        """Test ``validate_path`` raises PermissionError for a directory without read and execute permission."""
        with TemporaryDirectory() as tmp_dir:
            Path(tmp_dir).chmod(0o222)  # Remove read and execute permission
            with pytest.raises(PermissionError, match=r"Read or execute permissions denied for the path:*"):
                validate_path(tmp_dir, base_dir=Path(tmp_dir))

    @staticmethod
    def test_file_wrongsuffix() -> None:
        """Test ``validate_path`` raises ValueError for a file with wrong suffix."""
        with pytest.raises(ValueError, match=r"Path extension is not accepted."):
            validate_path("file.png", should_exist=False, extensions=(".json", ".txt"))

    @staticmethod
    def test_rejects_relative_escape_outside_base_dir() -> None:
        """Reject ``..`` paths that resolve outside ``base_dir``."""
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir) / "dataset"
            base.mkdir()
            secret = Path(tmp_dir) / "secret.txt"
            secret.write_text("secret", encoding="utf-8")
            with pytest.raises(ValueError, match="Access denied"):
                validate_path(base / ".." / "secret.txt", base_dir=base)

    @staticmethod
    def test_rejects_absolute_path_outside_base_dir() -> None:
        """Reject absolute paths outside ``base_dir``."""
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir) / "dataset"
            base.mkdir()
            outside = Path(tmp_dir) / "outside.txt"
            outside.write_text("x", encoding="utf-8")
            with pytest.raises(ValueError, match="Access denied"):
                validate_path(outside, base_dir=base)

    @staticmethod
    def test_allows_path_under_symlinked_base_dir() -> None:
        """Allow files under a dataset root that is a symlink to another folder."""
        with TemporaryDirectory() as tmp_dir:
            real_root = Path(tmp_dir) / "real_dataset"
            real_root.mkdir()
            target = real_root / "image.png"
            target.write_bytes(b"png")
            link_root = Path(tmp_dir) / "link_dataset"
            link_root.symlink_to(real_root, target_is_directory=True)
            validated = validate_path(link_root / "image.png", base_dir=link_root)
            assert validated == target.resolve()

    @staticmethod
    def test_rejects_escape_symlink_inside_base_dir() -> None:
        """Reject a symlink inside the tree that resolves outside ``base_dir``."""
        with TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir) / "dataset"
            base.mkdir()
            outside = Path(tmp_dir) / "outside.txt"
            outside.write_text("leaked", encoding="utf-8")
            escape_link = base / "escape.txt"
            escape_link.symlink_to(outside)
            with pytest.raises(ValueError, match="Access denied"):
                validate_path(escape_link, base_dir=base)

    @staticmethod
    def test_no_confinement_when_base_dir_omitted() -> None:
        """Omitting ``base_dir`` skips confinement (existence checks still apply)."""
        with TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "file.txt"
            file_path.write_text("ok", encoding="utf-8")
            assert validate_path(file_path) == file_path.resolve()


class TestResolvePathUnderRoot:
    """Tests for ``resolve_path_under_root``."""

    @staticmethod
    def test_joins_and_confines_relative_path() -> None:
        """Join a relative path under root and return the resolved path."""
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            nested = root / "a" / "b.png"
            nested.parent.mkdir(parents=True)
            nested.write_bytes(b"x")
            assert resolve_path_under_root(root, "a/b.png") == nested.resolve()

    @staticmethod
    def test_rejects_absolute_path_outside_root() -> None:
        """Reject absolute paths that resolve outside ``root``."""
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "root"
            root.mkdir()
            outside = Path(tmp_dir) / "file.txt"
            outside.write_text("x", encoding="utf-8")
            with pytest.raises(ValueError, match="Access denied"):
                resolve_path_under_root(root, outside)

    @staticmethod
    def test_allows_absolute_path_under_root() -> None:
        """Allow absolute paths that already resolve under ``root``."""
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            nested = root / "a" / "b.png"
            nested.parent.mkdir(parents=True)
            nested.write_bytes(b"x")
            assert resolve_path_under_root(root, nested) == nested.resolve()

    @staticmethod
    def test_rejects_parent_directory_escape() -> None:
        """Reject relative paths that escape root via ``..``."""
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir) / "root"
            root.mkdir()
            (Path(tmp_dir) / "secret.txt").write_text("secret", encoding="utf-8")
            with pytest.raises(ValueError, match="Access denied"):
                resolve_path_under_root(root, "../secret.txt", should_exist=False)


class TestGenerateOutputFilename:
    """Tests for ``generate_output_filename`` function."""

    @staticmethod
    def test_basic_mvtec_style() -> None:
        """Test basic MVTec-style dataset path handling."""
        input_path = Path("/data/MVTecAD/bottle/test/broken_large/000.png")
        output_path = Path("./results")
        result = generate_output_filename(input_path, output_path, "MVTecAD", "bottle")
        assert result == Path("./results/test/broken_large/000.png")

    @staticmethod
    def test_without_category() -> None:
        """Test path handling without category parameter."""
        input_path = Path("/data/MVTecAD/bottle/test/broken_large/000.png")
        output_path = Path("./results")
        result = generate_output_filename(input_path, output_path, "MVTecAD")
        assert result == Path("./results/bottle/test/broken_large/000.png")

    @staticmethod
    def test_folder_dataset() -> None:
        """Test handling of folder-based datasets."""
        input_path = Path("/datasets/MyDataset/normal/image001.png")
        output_path = Path("./output")
        result = generate_output_filename(input_path, output_path, "MyDataset")
        assert result == Path("./output/normal/image001.png")

    @staticmethod
    def test_custom_structure() -> None:
        """Test handling of custom directory structures."""
        input_path = Path("/custom/path/MyData/category/split/image.png")
        output_path = Path("./out")
        result = generate_output_filename(input_path, output_path, "MyData")
        assert result == Path("./out/category/split/image.png")

    @staticmethod
    def test_auto_detection() -> None:
        """Test auto-detection when dataset_name is not provided."""
        input_path = Path("/any/folder/structure/normal/image.png")
        output_path = Path("./out")
        result = generate_output_filename(input_path, output_path)
        assert result == Path("./out/normal/image.png")

    @staticmethod
    def test_mkdir_parameter() -> None:
        """Test mkdir parameter behavior."""
        with TemporaryDirectory() as tmp_dir:
            input_path = Path("/data/MVTecAD/bottle/test/000.png")
            output_path = Path(tmp_dir) / "results"

            # Test with mkdir=True (default)
            result = generate_output_filename(input_path, output_path, "MVTecAD")
            assert result.parent.exists()

            # Test with mkdir=False
            output_path = Path(tmp_dir) / "results2"
            result = generate_output_filename(input_path, output_path, "MVTecAD", mkdir=False)
            assert not result.parent.exists()

    @staticmethod
    def test_case_insensitive_matching() -> None:
        """Test case-insensitive matching of dataset and category names."""
        input_path = Path("/data/mvtecad/BOTTLE/test/000.png")
        output_path = Path("./results")
        result = generate_output_filename(input_path, output_path, "MVTecAD", "bottle")
        assert result == Path("./results/test/000.png")

    @staticmethod
    def test_relative_paths() -> None:
        """Test handling of relative input paths."""
        input_path = Path("data/MVTecAD/bottle/test/000.png")
        output_path = Path("./results")
        result = generate_output_filename(input_path, output_path, "MVTecAD")
        assert result == Path("./results/bottle/test/000.png")
