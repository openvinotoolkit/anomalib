"""Validate IO path data."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from pathlib import Path


def validate_path(path: str | Path | None) -> str | None:
    """Validate a single input path.

    Args:
        path: The input path to validate. Can be None, a string, or a Path object.

    Returns:
        - None if the input is None
        - A string representing the validated path

    Raises:
        TypeError: If the input is not None, a string, or a Path object.

    Examples:
        >>> validate_path(None)
        None
        >>> validate_path("/path/to/file.png")
        '/path/to/file.png'
        >>> from pathlib import Path
        >>> validate_path(Path("/path/to/file.png"))
        '/path/to/file.png'
    """
    if path is None:
        return None
    if isinstance(path, str | Path):
        return str(path)
    msg = f"Path must be None, a string, or Path object, got {type(path)}."
    raise TypeError(msg)


def validate_batch_path(
    paths: Sequence[str | Path] | None,
    batch_size: int | None = None,
) -> list[str] | None:
    """Validate a batch of input paths.

    Args:
        paths: A sequence of paths to validate, or None.
        batch_size: The expected number of paths. Defaults to None, in which case no batch size check is performed.

    Returns:
        - None if the input is None
        - A list of strings representing validated paths

    Raises:
        TypeError: If the input is not None or a sequence of strings or Path objects.
        ValueError: If a batch_size is specified and the number of paths doesn't match it.

    Examples:
        >>> paths = ["/path/to/file1.png", Path("/path/to/file2.png")]
        >>> validate_batch_path(paths, batch_size=2)
        ['/path/to/file1.png', '/path/to/file2.png']
        >>> validate_batch_path(paths)  # Without specifying batch_size
        ['/path/to/file1.png', '/path/to/file2.png']
        >>> validate_batch_path(paths, batch_size=3)
        Traceback (most recent call last):
            ...
        ValueError: Number of paths (2) does not match the specified batch size (3).
    """
    if paths is None:
        return None
    if not isinstance(paths, Sequence):
        msg = f"Paths must be None or a sequence of strings or Path objects, got {type(paths)}."
        raise TypeError(msg)
    if batch_size is not None and len(paths) != batch_size:
        msg = f"Number of paths ({len(paths)}) does not match the specified batch size ({batch_size})."
        raise ValueError(msg)

    validated_paths: list[str] = []
    for p in paths:
        if not isinstance(p, str | Path):
            msg = f"Each path in the sequence must be a string or Path object, got {type(p)}."
            raise TypeError(msg)
        validated_paths.append(str(p))
    return validated_paths
