"""Validate IO path data.

This module provides validation functions for file paths, ensuring they are in the
correct format and type.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from pathlib import Path


def validate_path(path: str | Path) -> str:
    """Validate a single input path.

    Args:
        path: Input path to validate. Can be a string or :class:`pathlib.Path`
            object.

    Returns:
        str: String representation of the validated path.

    Raises:
        TypeError: If ``path`` is not a string or :class:`pathlib.Path` object.

    Examples:
        >>> from pathlib import Path
        >>> validate_path("/path/to/file.png")
        '/path/to/file.png'
        >>> validate_path(Path("/path/to/file.png"))
        '/path/to/file.png'
        >>> validate_path(123)  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        TypeError: Path must be None, a string, or Path object, got <class 'int'>.
    """
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
        paths: Sequence of paths to validate. Each path can be a string or
            :class:`pathlib.Path` object. Can be ``None``.
        batch_size: Expected number of paths. If ``None``, no batch size check is
            performed.

    Returns:
        list[str] | None: List of validated path strings, or ``None`` if input is
        ``None``.

    Raises:
        TypeError: If ``paths`` is not ``None`` or a sequence of strings or
            :class:`pathlib.Path` objects.
        ValueError: If ``batch_size`` is specified and number of paths doesn't
            match it.

    Examples:
        >>> from pathlib import Path
        >>> paths = ["/path/to/file1.png", Path("/path/to/file2.png")]
        >>> validate_batch_path(paths, batch_size=2)
        ['/path/to/file1.png', '/path/to/file2.png']
        >>> validate_batch_path(paths)  # Without batch_size
        ['/path/to/file1.png', '/path/to/file2.png']
        >>> # With incorrect batch size
        >>> validate_batch_path(paths, batch_size=3)  # doctest: +IGNORE_EXCEPTION_DETAIL
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
