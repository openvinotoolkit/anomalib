"""Anomalib Path Utils."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path


def create_versioned_dir(root_dir: str | Path) -> Path:
    """Create a new version directory and update the ``latest`` symbolic link.

    Args:
        root_dir (Path): The root directory where the version directories are stored.

    Returns:
        latest_link_path (Path): The path to the ``latest`` symbolic link.

    Examples:
        >>> version_dir = create_version_dir(Path('path/to/experiments/'))
        PosixPath('/path/to/experiments/latest')

        >>> version_dir.resolve().name
        v1

        Calling the function again will create a new version directory and
        update the ``latest`` symbolic link:

        >>> version_dir = create_version_dir('path/to/experiments/')
        PosixPath('/path/to/experiments/latest')

        >>> version_dir.resolve().name
        v2

    """
    # Compile a regular expression to match version directories
    version_pattern = re.compile(r"^v(\d+)$")

    # Resolve the path
    root_dir = Path(root_dir).resolve()
    root_dir.mkdir(parents=True, exist_ok=True)

    # Find the highest existing version number
    highest_version = -1
    for version_dir in root_dir.iterdir():
        if version_dir.is_dir():
            match = version_pattern.match(version_dir.name)
            if match:
                version_number = int(match.group(1))
                highest_version = max(highest_version, version_number)

    # The new directory will have the next highest version number
    new_version_number = highest_version + 1
    new_version_dir = root_dir / f"v{new_version_number}"

    # Create the new version directory
    new_version_dir.mkdir()

    # Update the 'latest' symbolic link to point to the new version directory
    latest_link_path = root_dir / "latest"
    if latest_link_path.is_symlink() or latest_link_path.exists():
        latest_link_path.unlink()
    latest_link_path.symlink_to(new_version_dir, target_is_directory=True)

    return latest_link_path


def convert_to_snake_case(s: str) -> str:
    """Converts a string to snake case.

    Args:
        s (str): The input string to be converted.

    Returns:
        str: The converted string in snake case.

    Examples:
        >>> convert_to_snake_case("Snake Case")
        'snake_case'

        >>> convert_to_snake_case("snakeCase")
        'snake_case'

        >>> convert_to_snake_case("snake_case")
        'snake_case'
    """
    # Replace whitespace, hyphens, periods, and apostrophes with underscores
    s = re.sub(r"\s+|[-.\']", "_", s)

    # Insert underscores before capital letters (except at the beginning of the string)
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()

    # Remove leading and trailing underscores
    s = re.sub(r"^_+|_+$", "", s)

    # Replace multiple consecutive underscores with a single underscore
    return re.sub(r"__+", "_", s)
