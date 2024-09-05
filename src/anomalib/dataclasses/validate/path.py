"""Validate IO path data."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


def validate_path(path: str | Path) -> str:
    """Validate the input path.

    Args:
        path: The input path to validate. Can be a string or a Path object.

    Returns:
        The validated path as a string.

    Raises:
        TypeError: If the input is not a string or a Path object.

    Examples:
        >>> from anomalib.data.io.validate import validate_path

        >>> # String path
        >>> validate_path("/path/to/file.png")
        '/path/to/file.png'

        >>> # Path object
        >>> from pathlib import Path
        >>> validate_path(Path("/path/to/file.png"))
        '/path/to/file.png'

        >>> # Invalid input
        >>> validate_path(123)
        Traceback (most recent call last):
            ...
        TypeError: Path must be a string or a Path object, got <class 'int'>.
    """
    if isinstance(path, str | Path):
        return str(path)
    msg = f"Path must be a string or a Path object, got {type(path)}."
    raise TypeError(msg)
