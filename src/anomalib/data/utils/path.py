# Copyright (C) 2022-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Path utilities for handling file paths in anomalib.

This module provides utilities for:

- Validating and resolving file paths
- Checking path length and character restrictions
- Converting between path types
- Handling file extensions
- Managing directory types for anomaly detection
- Confining paths under an allowed base directory

Example:
    >>> from anomalib.data.utils.path import validate_path
    >>> path = validate_path("./datasets/MVTecAD/bottle/train/good/000.png")
    >>> print(path)
    PosixPath('/abs/path/to/anomalib/datasets/MVTecAD/bottle/train/good/000.png')

    >>> from anomalib.data.utils.path import DirType
    >>> print(DirType.NORMAL)
    normal
"""

import logging
import os
import re
from enum import Enum
from pathlib import Path

from torchvision.datasets.folder import IMG_EXTENSIONS

logger = logging.getLogger(__name__)


def is_within_directory(directory: Path, target: Path) -> bool:
    """Check if a target path is located within a given directory.

    Both paths are resolved (following symlinks) before comparison, so a
    dataset root that is itself a symlink to another folder or drive is
    allowed, while escape symlinks or ``..`` segments that resolve outside
    ``directory`` are rejected.

    Args:
        directory: Path of the parent directory
        target: Path to check

    Returns:
        ``True`` if target is within directory, ``False`` otherwise
    """
    abs_directory = directory.resolve()
    abs_target = target.resolve()

    return abs_target.is_relative_to(abs_directory)


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

    # convert extensions to lowercase for case-insensitive matching
    extensions = tuple(ext.lower() for ext in extensions)

    if isinstance(extensions, str):
        extensions = (extensions,)

    if not all(extension.startswith(".") for extension in extensions):
        msg = f"All extensions {extensions} must start with the dot"
        raise RuntimeError(msg)

    filenames = [f for f in path.glob("**/*") if f.suffix.lower() in extensions and not f.is_dir()]
    # list of files that are in hidden directories or are hidden files themselves
    hidden_files = [f for f in filenames if any(part.startswith(".") for part in f.parts)]
    if hidden_files:
        logger.warning(f"{len(hidden_files)} hidden files found in {path}. Please make sure this is intended.")

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
    """Validate path for existence, permissions, extension and confinement.

    Args:
        path: Path to validate
        base_dir: If provided, restrict access to paths that resolve under
            this directory. When ``None``, no confinement check is applied.
        should_exist: If ``True``, verify path exists
        extensions: Allowed file extensions

    Returns:
        Validated Path object

    Raises:
        TypeError: If path is invalid type
        ValueError: If path is too long, has invalid characters/extension, or
            resolves outside ``base_dir``
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

    # Sanitize path (follows symlinks)
    path = Path(path).resolve()

    # Confine to base_dir when requested
    if base_dir is not None:
        resolved_base = Path(base_dir).resolve()
        if not is_within_directory(resolved_base, path):
            msg = "Access denied: Path is outside the allowed directory"
            raise ValueError(msg)

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


def resolve_path_under_root(
    root: str | Path,
    path: str | Path,
    *,
    should_exist: bool = True,
    extensions: tuple[str, ...] | None = None,
) -> Path:
    """Resolve ``path`` under ``root`` and confine the result to ``root``.

    Relative paths are joined with ``root``. Absolute paths are kept as-is but
    must still resolve under ``root`` (so ``Path(root, absolute)`` cannot bypass
    confinement on POSIX).

    Args:
        root: Dataset or output root that paths must remain under
        path: Path relative to ``root``, or an absolute path under ``root``
        should_exist: If ``True``, verify the resolved path exists
        extensions: Allowed file extensions

    Returns:
        Validated absolute path under ``root``

    Raises:
        ValueError: If ``path`` resolves outside ``root``
    """
    root_path = Path(root)
    path = Path(path)
    candidate = path if path.is_absolute() else root_path / path
    return validate_path(
        candidate,
        base_dir=root_path,
        should_exist=should_exist,
        extensions=extensions,
    )


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
