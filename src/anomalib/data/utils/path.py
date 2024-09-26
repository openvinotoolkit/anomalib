"""Path Utils."""

# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import re
from enum import Enum
from pathlib import Path

from torchvision.datasets.folder import IMG_EXTENSIONS


class DirType(str, Enum):
    """Dir type names."""

    NORMAL = "normal"
    ABNORMAL = "abnormal"
    NORMAL_TEST = "normal_test"
    NORMAL_DEPTH = "normal_depth"
    ABNORMAL_DEPTH = "abnormal_depth"
    NORMAL_TEST_DEPTH = "normal_test_depth"
    MASK = "mask_dir"


def _check_and_convert_path(path: str | Path) -> Path:
    """Check an input path, and convert to Pathlib object.

    Args:
        path (str | Path): Input path.

    Returns:
        Path: Output path converted to pathlib object.
    """
    if not isinstance(path, Path):
        path = Path(path)
    return path


def _prepare_files_labels(
    path: str | Path,
    path_type: str,
    extensions: tuple[str, ...] | None = None,
) -> tuple[list, list]:
    """Return a list of filenames and list corresponding labels.

    Args:
        path (str | Path): Path to the directory containing images.
        path_type (str): Type of images in the provided path ("normal", "abnormal", "normal_test")
        extensions (tuple[str, ...] | None, optional): Type of the image extensions to read from the
            directory.

    Returns:
        List, List: Filenames of the images provided in the paths, labels of the images provided in the paths
    """
    path = _check_and_convert_path(path)
    if extensions is None:
        extensions = IMG_EXTENSIONS

    if isinstance(extensions, str):
        extensions = (extensions,)

    if not all(extension.startswith(".") for extension in extensions):
        msg = f"All extensions {extensions} must start with the dot"
        raise RuntimeError(msg)

    filenames = [
        f
        for f in path.glob("**/*")
        if f.suffix in extensions and not f.is_dir() and not any(part.startswith(".") for part in f.parts)
    ]
    if not filenames:
        msg = f"Found 0 {path_type} images in {path} with extensions {extensions}"
        raise RuntimeError(msg)

    labels = [path_type] * len(filenames)

    return filenames, labels


def resolve_path(folder: str | Path, root: str | Path | None = None) -> Path:
    """Combine root and folder and returns the absolute path.

    This allows users to pass either a root directory and relative paths, or absolute paths to each of the
    image sources. This function makes sure that the samples dataframe always contains absolute paths.

    Args:
        folder (str | Path | None): Folder location containing image or mask data.
        root (str | Path | None): Root directory for the dataset.
    """
    folder = Path(folder)
    if folder.is_absolute():
        path = folder
    # path is relative.
    elif root is None:
        # no root provided; return absolute path
        path = folder.resolve()
    else:
        # root provided; prepend root and return absolute path
        path = (Path(root) / folder).resolve()
    return path


def is_path_too_long(path: str | Path, max_length: int = 512) -> bool:
    r"""Check if the path contains too long input.

    Args:
        path (str | Path): Path to check.
        max_length (int): Maximum length a path can be before it is considered too long.
            Defaults to ``512``.

    Returns:
        bool: True if the path contains too long input, False otherwise.

    Examples:
        >>> contains_too_long_input("./datasets/MVTec/bottle/train/good/000.png")
        False

        >>> contains_too_long_input("./datasets/MVTec/bottle/train/good/000.png" + "a" * 4096)
        True
    """
    return len(str(path)) > max_length


def contains_non_printable_characters(path: str | Path) -> bool:
    r"""Check if the path contains non-printable characters.

    Args:
        path (str | Path): Path to check.

    Returns:
        bool: True if the path contains non-printable characters, False otherwise.

    Examples:
        >>> contains_non_printable_characters("./datasets/MVTec/bottle/train/good/000.png")
        False

        >>> contains_non_printable_characters("./datasets/MVTec/bottle/train/good/000.png\0")
        True
    """
    printable_pattern = re.compile(r"^[\x20-\x7E]+$")
    return not printable_pattern.match(str(path))


def validate_path(
    path: str | Path,
    base_dir: str | Path | None = None,
    should_exist: bool = True,
    extensions: tuple[str, ...] | None = None,
) -> Path:
    """Validate the path.

    Args:
        path (str | Path): Path to validate.
        base_dir (str | Path): Base directory to restrict file access.
        should_exist (bool): If True, do not raise an exception if the path does not exist.
        extensions (tuple[str, ...] | None): Accepted extensions for the path. An exception is raised if the
            path does not have one of the accepted extensions. If None, no check is performed. Defaults to None.

    Returns:
        Path: Validated path.

    Examples:
        >>> validate_path("./datasets/MVTec/bottle/train/good/000.png")
        PosixPath('/abs/path/to/anomalib/datasets/MVTec/bottle/train/good/000.png')

        >>> validate_path("./datasets/MVTec/bottle/train/good/000.png", base_dir="./datasets/MVTec")
        PosixPath('/abs/path/to/anomalib/datasets/MVTec/bottle/train/good/000.png')

        >>> validate_path("/path/to/unexisting/file")
        Traceback (most recent call last):
        File "<string>", line 1, in <module>
        File "<string>", line 18, in validate_path
        FileNotFoundError: Path does not exist: /path/to/unexisting/file

        Accessing a file without read permission should raise PermissionError:

        .. note::

            Note that, we are using ``/usr/local/bin`` directory as an example here.
            If this directory does not exist on your system, this will raise
            ``FileNotFoundError`` instead of ``PermissionError``. You could change
            the directory to any directory that you do not have read permission.

        >>> validate_path("/bin/bash", base_dir="/bin/")
        Traceback (most recent call last):
        File "<string>", line 1, in <module>
        File "<string>", line 18, in validate_path
        PermissionError: Read permission denied for the file: /usr/local/bin

    """
    # Check if the path is of an appropriate type
    if not isinstance(path, str | Path):
        raise TypeError("Expected str, bytes or os.PathLike object, not " + type(path).__name__)

    # Check if the path is too long
    if is_path_too_long(path):
        msg = f"Path is too long: {path}"
        raise ValueError(msg)

    # Check if the path contains non-printable characters
    if contains_non_printable_characters(path):
        msg = f"Path contains non-printable characters: {path}"
        raise ValueError(msg)

    # Sanitize paths
    path = Path(path).resolve()
    base_dir = Path(base_dir).resolve() if base_dir else Path.home()

    # In case path ``should_exist``, the path is valid, and should be
    # checked for read and execute permissions.
    if should_exist:
        # Check if the path exists
        if not path.exists():
            msg = f"Path does not exist: {path}"
            raise FileNotFoundError(msg)

        # Check the read and execute permissions
        if not (os.access(path, os.R_OK) or os.access(path, os.X_OK)):
            msg = f"Read or execute permissions denied for the path: {path}"
            raise PermissionError(msg)

    # Check if the path has one of the accepted extensions
    if extensions is not None and path.suffix not in extensions:
        msg = f"Path extension is not accepted. Accepted extensions: {extensions}. Path: {path}"
        raise ValueError(msg)

    return path


def validate_and_resolve_path(
    folder: str | Path,
    root: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> Path:
    """Validate and resolve the path.

    Args:
        folder (str | Path): Folder location containing image or mask data.
        root (str | Path | None): Root directory for the dataset.
        base_dir (str | Path | None): Base directory to restrict file access.

    Returns:
        Path: Validated and resolved path.
    """
    return validate_path(resolve_path(folder, root), base_dir)
