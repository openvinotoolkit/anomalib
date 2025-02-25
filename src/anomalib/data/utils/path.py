"""Path utilities for handling file paths in anomalib.

This module provides utilities for:

- Validating and resolving file paths
- Checking path length and character restrictions
- Converting between path types
- Handling file extensions
- Managing directory types for anomaly detection

Example:
    >>> from anomalib.data.utils.path import validate_path
    >>> path = validate_path("./datasets/MVTecAD/bottle/train/good/000.png")
    >>> print(path)
    PosixPath('/abs/path/to/anomalib/datasets/MVTecAD/bottle/train/good/000.png')

    >>> from anomalib.data.utils.path import DirType
    >>> print(DirType.NORMAL)
    normal
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import re
from enum import Enum
from pathlib import Path

from torchvision.datasets.folder import IMG_EXTENSIONS


class DirType(str, Enum):
    """Directory type names for organizing anomaly detection datasets.

    Attributes:
        NORMAL: Directory containing normal/good samples for training
        ABNORMAL: Directory containing anomalous/defective samples
        NORMAL_TEST: Directory containing normal test samples
        NORMAL_DEPTH: Directory containing depth maps for normal samples
        ABNORMAL_DEPTH: Directory containing depth maps for abnormal samples
        NORMAL_TEST_DEPTH: Directory containing depth maps for normal test samples
        MASK: Directory containing ground truth segmentation masks
    """

    NORMAL = "normal"
    ABNORMAL = "abnormal"
    NORMAL_TEST = "normal_test"
    NORMAL_DEPTH = "normal_depth"
    ABNORMAL_DEPTH = "abnormal_depth"
    NORMAL_TEST_DEPTH = "normal_test_depth"
    MASK = "mask_dir"


def _check_and_convert_path(path: str | Path) -> Path:
    """Check and convert input path to pathlib object.

    Args:
        path: Input path as string or Path object

    Returns:
        Path object of the input path

    Example:
        >>> path = _check_and_convert_path("./datasets/example.png")
        >>> isinstance(path, Path)
        True
    """
    if not isinstance(path, Path):
        path = Path(path)
    return path


def _prepare_files_labels(
    path: str | Path,
    path_type: str,
    extensions: tuple[str, ...] | None = None,
) -> tuple[list, list]:
    """Get lists of filenames and corresponding labels from a directory.

    Args:
        path: Path to directory containing images
        path_type: Type of images ("normal", "abnormal", "normal_test")
        extensions: Allowed file extensions. Defaults to ``IMG_EXTENSIONS``

    Returns:
        Tuple containing:
            - List of image filenames
            - List of corresponding labels

    Raises:
        RuntimeError: If no valid images found or extensions don't start with dot

    Example:
        >>> files, labels = _prepare_files_labels("./normal", "normal", (".png",))
        >>> len(files) == len(labels)
        True
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
    """Combine root and folder paths into absolute path.

    Args:
        folder: Folder location containing image or mask data
        root: Optional root directory for the dataset

    Returns:
        Absolute path combining root and folder

    Example:
        >>> path = resolve_path("subdir", "/root")
        >>> path.is_absolute()
        True
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
    """Check if path exceeds maximum allowed length.

    Args:
        path: Path to check
        max_length: Maximum allowed path length. Defaults to ``512``

    Returns:
        ``True`` if path is too long, ``False`` otherwise

    Example:
        >>> is_path_too_long("short_path.txt")
        False
        >>> is_path_too_long("a" * 1000)
        True
    """
    return len(str(path)) > max_length


def contains_non_printable_characters(path: str | Path) -> bool:
    r"""Check if path contains non-printable characters.

    Args:
        path: Path to check

    Returns:
        ``True`` if path contains non-printable chars, ``False`` otherwise

    Example:
        >>> contains_non_printable_characters("normal.txt")
        False
        >>> contains_non_printable_characters("test\x00.txt")
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
    """Validate path for existence, permissions and extension.

    Args:
        path: Path to validate
        base_dir: Base directory to restrict file access
        should_exist: If ``True``, verify path exists
        extensions: Allowed file extensions

    Returns:
        Validated Path object

    Raises:
        TypeError: If path is invalid type
        ValueError: If path is too long or has invalid characters/extension
        FileNotFoundError: If path doesn't exist when required
        PermissionError: If path lacks required permissions

    Example:
        >>> path = validate_path("./datasets/image.png", extensions=(".png",))
        >>> path.suffix
        '.png'
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
        msg = f"Path extension is not accepted. Accepted: {extensions}. Path: {path}"
        raise ValueError(msg)

    return path


def validate_and_resolve_path(
    folder: str | Path,
    root: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> Path:
    """Validate and resolve path by combining validation and resolution.

    Args:
        folder: Folder location containing image or mask data
        root: Root directory for the dataset
        base_dir: Base directory to restrict file access

    Returns:
        Validated and resolved absolute Path

    Example:
        >>> path = validate_and_resolve_path("subdir", "/root")
        >>> path.is_absolute()
        True
    """
    return validate_path(resolve_path(folder, root), base_dir)
