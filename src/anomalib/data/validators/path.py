"""Validate IO path data.

This module provides validators for file system paths. The validators ensure path
consistency and correctness.

The validators check:
    - Path types (str vs Path objects)
    - Path string formatting
    - Batch size consistency
    - None handling

Example:
    Validate a single path::

        >>> from anomalib.data.validators import validate_path
        >>> path = "/path/to/file.jpg"
        >>> validated = validate_path(path)
        >>> validated == path
        True

    Validate a batch of paths::

        >>> from anomalib.data.validators import validate_batch_path
        >>> paths = ["/path/1.jpg", "/path/2.jpg"]
        >>> validated = validate_batch_path(paths, batch_size=2)
        >>> len(validated)
        2

Note:
    The validators are used internally by the data modules to ensure path
    consistency before processing.
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from pathlib import Path


def validate_path(path: str | Path) -> str:
    """Validate a single input path.

    This function validates and normalizes file system paths. It accepts string paths or
    ``pathlib.Path`` objects and converts them to string format.

    Args:
        path (``str`` | ``Path``): Input path to validate. Can be a string path or
            ``pathlib.Path`` object.

    Returns:
        ``str``: The validated path as a string.

    Raises:
        TypeError: If ``path`` is not a string or ``Path`` object.

    Examples:
        Validate a string path::

            >>> validate_path("/path/to/file.png")
            '/path/to/file.png'

        Validate a Path object::

            >>> from pathlib import Path
            >>> validate_path(Path("/path/to/file.png"))
            '/path/to/file.png'

        Invalid input raises TypeError::

            >>> validate_path(123)
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

    This function validates and normalizes a sequence of file system paths. It accepts a
    sequence of string paths or ``pathlib.Path`` objects and converts them to a list of
    string paths. Optionally checks if the number of paths matches an expected batch size.

    Args:
        paths (``Sequence[str | Path] | None``): A sequence of paths to validate, or
            ``None``. Each path can be a string or ``pathlib.Path`` object.
        batch_size (``int | None``, optional): The expected number of paths. If specified,
            validates that the number of paths matches this value. Defaults to ``None``,
            in which case no batch size check is performed.

    Returns:
        ``list[str] | None``: A list of validated paths as strings, or ``None`` if the
        input is ``None``.

    Raises:
        TypeError: If ``paths`` is not ``None`` or a sequence of strings/``Path`` objects.
        ValueError: If ``batch_size`` is specified and the number of paths doesn't match.

    Examples:
        Validate a list of paths with batch size check::

            >>> from pathlib import Path
            >>> paths = ["/path/to/file1.png", Path("/path/to/file2.png")]
            >>> validate_batch_path(paths, batch_size=2)
            ['/path/to/file1.png', '/path/to/file2.png']

        Validate without batch size check::

            >>> validate_batch_path(paths)  # Without specifying batch_size
            ['/path/to/file1.png', '/path/to/file2.png']

        Batch size mismatch raises ValueError::

            >>> validate_batch_path(paths, batch_size=3)
            Traceback (most recent call last):
                ...
            ValueError: Number of paths (2) does not match the specified batch size (3).

        Invalid input type raises TypeError::

            >>> validate_batch_path("not_a_sequence")
            Traceback (most recent call last):
                ...
            TypeError: Paths must be None or a sequence of strings or Path objects...
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
