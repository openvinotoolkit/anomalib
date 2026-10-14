# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Path utilities for anomaly detection.

This module provides utilities for managing paths and directories in anomaly
detection projects. The key components include:

    - Pre-trained weights cache directory management
    - Version directory creation and management
    - Symbolic link handling
    - Path resolution and validation
    - Output filename generation

Examples:
    Get the platform-appropriate cache directory for pre-trained weights:

    >>> from anomalib.utils.path import get_pretrained_weights_dir
    >>> weights_dir = get_pretrained_weights_dir()

    Test create_versioned_dir:

    >>> from anomalib.utils.path import create_versioned_dir
    >>> from pathlib import Path
    >>> # Create versioned directory
    >>> version_dir = create_versioned_dir(Path("experiments"))
    >>> version_dir.name
    'v1'

The module ensures consistent path handling by:
    - Providing a centralized, cross-platform cache path for pre-trained weights
    - Creating incrementing version directories (v1, v2, etc.)
    - Maintaining a ``latest`` symbolic link
    - Handling both string and ``Path`` inputs
    - Providing cross-platform compatibility

Note:
    All paths are resolved to absolute paths to ensure consistent behavior
    across different working directories.
"""

import logging
import os
import re
import shutil
import sys
from contextlib import suppress
from pathlib import Path, PurePath
from typing import TypeAlias

import platformdirs

logger = logging.getLogger(__name__)

PathValue: TypeAlias = (
    str | int | float | bool | PurePath | list["PathValue"] | tuple["PathValue", ...] | dict[str, "PathValue"] | None
)


def paths_to_strings(value: PathValue) -> PathValue:
    """Convert paths nested in checkpoint data to strings.

    Args:
        value: Checkpoint data containing zero or more path values.

    Returns:
        Equivalent data with all path values converted to strings.

    Example:
        >>> paths_to_strings({"weights": Path("models/model.ckpt")})
        {'weights': 'models/model.ckpt'}
    """
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, dict):
        return {key: paths_to_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [paths_to_strings(item) for item in value]
    if isinstance(value, tuple):
        return tuple(paths_to_strings(item) for item in value)
    return value


def _get_cache_subdir(subdir: str) -> Path:
    """Return a subdirectory under the platform-appropriate anomalib cache root.

    Uses ``platformdirs`` to resolve the user cache directory, then appends
    ``anomalib/<subdir>``. Both the root and subdirectory are created if they
    do not already exist.

    The *subdir* argument is validated to prevent path traversal attacks:
    it must be a simple relative path without ``..`` components, and it must
    not be absolute.

    Args:
        subdir (str): Relative path component(s) under the anomalib cache root
            (e.g. ``"datasets"`` or ``"datasets/MVTecAD"``).

    Returns:
        Path: Absolute path to the requested cache subdirectory.

    Raises:
        ValueError: If *subdir* is empty, absolute, or contains ``..`` components.
    """
    subdir_path = Path(subdir)
    if not subdir or subdir_path.is_absolute() or ".." in subdir_path.parts:
        msg = f"Invalid cache subdirectory name: {subdir!r}. Must be a relative path without '..' components."
        raise ValueError(msg)

    cache_dir = platformdirs.user_cache_path("anomalib", ensure_exists=True) / subdir_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_pretrained_weights_dir() -> Path:
    """Return the platform-appropriate cache directory for pre-trained model weights.

    Returns:
        Path: Path to the pre-trained weights cache directory.

    Example:
        >>> from anomalib.utils.path import get_pretrained_weights_dir
        >>> get_pretrained_weights_dir().name
        'pre_trained'
        >>> get_pretrained_weights_dir()  # linux
        PosixPath('/home/user/.cache/anomalib/pre_trained')
    """
    return _get_cache_subdir("pre_trained")


def get_datasets_dir() -> Path:
    """Return the platform-appropriate cache directory for anomalib datasets.

    Returns:
        Path: Path to the datasets cache directory.

    Example:
        >>> from anomalib.utils.path import get_datasets_dir
        >>> get_datasets_dir().name
        'datasets'
        >>> get_datasets_dir()  # linux
        PosixPath('/home/user/.cache/anomalib/datasets')
    """
    return _get_cache_subdir("datasets")


def resolve_dataset_root(root: str | Path | None, dataset_name: str) -> Path:
    """Resolve a dataset root path.

    Args:
        root: Root directory of the dataset. When ``None``, uses the anomalib
            datasets cache directory for ``dataset_name``.
        dataset_name: Name of the dataset. Used only when ``root`` is ``None``.

    Returns:
        Path: Resolved dataset root path.
    """
    return Path(root) if root is not None else get_datasets_dir() / dataset_name


def _highest_version_dir(parent: Path) -> str | None:
    """Return the highest version directory name (e.g. 'v2') under parent, or None.

    Scans parent for directories matching v0, v1, v2, ... and returns the name of the one with the
    highest number. Used by create_versioned_dir and resolve_versioned_path to avoid duplication.

    Args:
        parent (Path): Directory that may contain versioned subdirs (v0, v1, ...).

    Returns:
        str | None: Name of the highest version dir (e.g. 'v2'), or None if none exist.
    """
    try:
        if not parent.is_dir():
            return None

        highest = -1
        version_pattern = re.compile(r"^v(\d+)$")
        for child in parent.iterdir():
            if child.is_dir() and (match := version_pattern.match(child.name)):
                highest = max(highest, int(match.group(1)))
    except OSError:
        return None
    return f"v{highest}" if highest >= 0 else None


def _validate_windows_path(path: Path) -> bool:
    """Validate that a path is safe for use in Windows commands.

    Args:
        path: Path to validate

    Returns:
        True if path is safe, False otherwise
    """
    path_str = str(path)

    # Check for shell metacharacters that could be dangerous
    dangerous_chars = {"&", "|", ";", "<", ">", "^", '"', "'", "`", "$", "(", ")", "*", "?", "[", "]", "{", "}"}

    # Perform all validation checks
    if (
        any(char in path_str for char in dangerous_chars)
        or "\x00" in path_str
        # Traditional Windows MAX_PATH is 260 characters. This is a conservative
        # limit and does not take optional long-path support into account.
        or (sys.platform.startswith("win") and len(path_str) > 260)
    ):
        return False

    # Ensure the path exists and is actually a directory (for target)
    # or that its parent exists (for tmp)
    try:
        return path.is_dir() if path.exists() else path.parent.exists()
    except (OSError, ValueError):
        return False


def _is_windows_junction(p: Path) -> bool:
    """Return True if path is a directory junction."""
    if not sys.platform.startswith("win"):
        return False

    try:
        # On Windows, check if it's a directory that's not a symlink
        # Junctions appear as directories but resolve to different paths
        return p.exists() and p.is_dir() and not p.is_symlink() and p.resolve() != p
    except (OSError, RuntimeError):
        # Handle cases where path operations fail
        return False


def _safe_remove_path(p: Path) -> None:
    """Remove file/dir/symlink/junction at p without following links."""
    if not os.path.lexists(str(p)):
        return
    with suppress(FileNotFoundError):
        if p.is_symlink():
            p.unlink()
        elif _is_windows_junction(p):
            # Use rmdir for Windows junctions
            p.rmdir()
        elif p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()


def _make_latest_windows(latest: Path, target: Path) -> None:
    # Clean previous latest (symlink/junction/dir/file)
    _safe_remove_path(latest)

    tmp = latest.with_name(latest.name + "_tmp")
    _safe_remove_path(tmp)

    # Try creating a directory junction using native Python API
    try:
        # Use Path.symlink_to with target_is_directory=True for directory junction on Windows
        # This creates a junction point that doesn't require admin privileges
        tmp.symlink_to(target.resolve(), target_is_directory=True)
    except (OSError, NotImplementedError):
        # Try using Windows mklink command via subprocess
        try:
            import subprocess  # nosec B404

            # Note: Using subprocess with mklink is safe here as we control
            # the command and arguments. This is a standard Windows command.
            if not _validate_windows_path(tmp) or not _validate_windows_path(target):
                logger.warning(
                    "Warning: Unsafe characters detected in paths. Falling back to text pointer file for 'latest'.",
                )
                msg = f"Unsafe path detected: {tmp} -> {target}"
                raise ValueError(msg)
            cmd_exe = shutil.which("cmd.exe")
            if not cmd_exe:
                msg = "Cannot locate cmd.exe on PATH"
                raise OSError(msg)  # noqa: TRY301
            result = subprocess.run(  # noqa: S603  # nosec B603
                [
                    cmd_exe,
                    "/c",
                    "mklink",
                    "/J",
                    str(tmp),
                    str(target.resolve()),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0 and tmp.exists():
                tmp.replace(latest)
                return
            logger.warning(
                "Failed to create Windows junction with mklink (return code %s). Command: %s, stderr: %r",
                result.returncode,
                result.args,
                result.stderr,
            )
        except (subprocess.SubprocessError, OSError):
            # Subprocess failed; intentionally fall through to the final
            # fallback below that creates a text pointer file.
            logger.debug(
                "Failed to create Windows junction using mklink; falling back to pointer file.",
                exc_info=True,
            )
    else:
        # Only reached if symlink creation succeeded
        tmp.replace(latest)
        return

    # Final fallback: create a text file indicating the latest version
    # This preserves the intended behavior without breaking the system
    latest.mkdir(exist_ok=True)
    version_file = latest / ".version_pointer"
    version_file.write_text(str(target.resolve()))


def create_versioned_dir(root_dir: str | Path) -> Path:
    """Create a new version directory and update the ``latest`` symbolic link.

    This function creates a new versioned directory (e.g. ``v1``, ``v2``, etc.) inside the
    specified root directory and updates a ``latest`` symbolic link to point to it.
    The version numbers increment automatically based on existing directories.

    Args:
        root_dir (Union[str, Path]): Root directory path where version directories will be
            created. Can be provided as a string or ``Path`` object. Directory will be
            created if it doesn't exist.

    Returns:
        Path: Path to the ``latest`` symbolic link that points to the newly created
            version directory.

    Examples:
        Create first version directory:

        >>> from pathlib import Path
        >>> version_dir = create_versioned_dir(Path("experiments"))
        >>> version_dir
        PosixPath('experiments/latest')
        >>> version_dir.resolve().name  # Points to v1
        'v1'

        Create second version directory:

        >>> version_dir = create_versioned_dir("experiments")
        >>> version_dir.resolve().name  # Now points to v2
        'v2'

    Note:
        - The function resolves all paths to absolute paths
        - Creates parent directories if they don't exist
        - Handles existing symbolic links by removing and recreating them
        - Version directories follow the pattern ``v1``, ``v2``, etc.
        - The ``latest`` link always points to the most recently created version
    """
    root_dir = Path(root_dir).resolve()
    root_dir.mkdir(parents=True, exist_ok=True)

    highest = _highest_version_dir(root_dir)
    next_num = int(highest[1:]) + 1 if highest else 0
    new_version_dir = root_dir / f"v{next_num}"

    # Create the new version directory
    new_version_dir.mkdir()

    # Update the 'latest' symbolic link to point to the new version directory
    latest_link_path = root_dir / "latest"
    if sys.platform.startswith("win"):
        _make_latest_windows(latest_link_path, new_version_dir)
    else:
        if latest_link_path.is_symlink() or latest_link_path.exists():
            latest_link_path.unlink()
        latest_link_path.symlink_to(new_version_dir, target_is_directory=True)

    # Return the versioned directory path, not the latest link
    # This ensures training saves to the versioned directory directly
    return new_version_dir


def resolve_versioned_path(path: str | Path) -> Path:
    """Resolve a path by replacing a ``latest`` component with the actual version dir.

    If the path contains a component named ``latest`` (e.g. from a symlink or junction used by
    ``create_versioned_dir``), returns the concrete path. On POSIX, the symlink is followed via
    ``Path.resolve()``. On Windows, traversing the junction can raise WinError 448 (untrusted
    mount point), so the path is resolved by replacing ``latest`` with the highest version dir
    (v0, v1, ...) using ``_highest_version_dir`` without traversing the junction.

    Args:
        path (str | Path): Path that may contain ``latest`` as a component (e.g.
            ``.../latest/weights/lightning/model.ckpt``).

    Returns:
        Path: Resolved path (symlink followed on POSIX; ``latest`` replaced by actual version on
            Windows), or the original path if no ``latest`` component or resolution not possible.

    Example:
        >>> from pathlib import Path
        >>> # If /exp contains v0, v1 and a 'latest' link to v1:
        >>> resolve_versioned_path(Path("/exp/latest/weights/model.ckpt"))
        PosixPath('/exp/v1/weights/model.ckpt')
    """
    path = Path(path)
    result = path
    if "latest" in (parts := list(path.parts)):
        if sys.platform != "win32":
            # POSIX: follow the symlink directly
            try:
                result = path.resolve()
            except OSError:
                result = path
        else:
            # Windows: avoid traversing the junction; replace "latest" with highest vN.
            idx = parts.index("latest")
            parent = Path(*parts[:idx])
            version_dir = _highest_version_dir(parent)
            if version_dir:
                parts[idx] = version_dir
                result = Path(*parts)
    return result


def convert_to_snake_case(s: str) -> str:
    """Convert a string to snake case format.

    This function converts various string formats (space-separated, camelCase,
    PascalCase, etc.) to snake_case by:

    - Converting spaces and punctuation to underscores
    - Inserting underscores before capital letters
    - Converting to lowercase
    - Removing redundant underscores

    Args:
        s (str): Input string to convert to snake case.

    Returns:
        str: The input string converted to snake case format.

    Examples:
        Convert space-separated string:

        >>> convert_to_snake_case("Snake Case")
        'snake_case'

        Convert camelCase:

        >>> convert_to_snake_case("snakeCase")
        'snake_case'

        Convert PascalCase:

        >>> convert_to_snake_case("SnakeCase")
        'snake_case'

        Handle existing snake_case:

        >>> convert_to_snake_case("snake_case")
        'snake_case'

        Handle punctuation:

        >>> convert_to_snake_case("snake.case")
        'snake_case'

        >>> convert_to_snake_case("snake-case")
        'snake_case'

    Note:
        - Leading/trailing underscores are removed
        - Multiple consecutive underscores are collapsed to a single underscore
        - Punctuation marks (``.``, ``-``, ``'``) are converted to underscores
    """
    # Replace whitespace, hyphens, periods, and apostrophes with underscores
    s = re.sub(r"\s+|[-.\']", "_", s)

    # Insert underscores before capital letters (except at the beginning of the string)
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()

    # Remove leading and trailing underscores
    s = re.sub(r"^_+|_+$", "", s)

    # Replace multiple consecutive underscores with a single underscore
    return re.sub(r"__+", "_", s)


def convert_snake_to_pascal_case(snake_case: str) -> str:
    """Convert snake_case string to PascalCase.

    This function takes a string in snake_case format (words separated by underscores)
    and converts it to PascalCase format (each word capitalized and concatenated).

    Args:
        snake_case (str): Input string in snake_case format (e.g. ``"efficient_ad"``)

    Returns:
        str: Output string in PascalCase format (e.g. ``"EfficientAd"``)

    Examples:
        >>> convert_snake_to_pascal_case("efficient_ad")
        'EfficientAd'
        >>> convert_snake_to_pascal_case("patchcore")
        'Patchcore'
        >>> convert_snake_to_pascal_case("reverse_distillation")
        'ReverseDistillation'
    """
    return "".join(word.capitalize() for word in snake_case.split("_"))


def convert_to_title_case(text: str) -> str:
    """Convert text to title case, handling various text formats.

    This function converts text from various formats (regular text, snake_case, camelCase,
    PascalCase) to title case format. It preserves punctuation and handles contractions
    appropriately.

    Args:
        text (str): Input text to convert to title case. Can be in any text format like
            snake_case, camelCase, PascalCase or regular text.

    Returns:
        str: The input text converted to title case format.

    Raises:
        TypeError: If the input ``text`` is not a string.

    Examples:
        Regular text:

        >>> convert_to_title_case("the quick brown fox")
        'The Quick Brown Fox'

        Snake case:

        >>> convert_to_title_case("convert_snake_case_to_title_case")
        'Convert Snake Case To Title Case'

        Camel case:

        >>> convert_to_title_case("convertCamelCaseToTitleCase")
        'Convert Camel Case To Title Case'

        Pascal case:

        >>> convert_to_title_case("ConvertPascalCaseToTitleCase")
        'Convert Pascal Case To Title Case'

        Mixed cases:

        >>> convert_to_title_case("mixed_snake_camelCase and PascalCase")
        'Mixed Snake Camel Case And Pascal Case'

        Handling punctuation and contractions:

        >>> convert_to_title_case("what's the_weather_like? it'sSunnyToday.")
        "What's The Weather Like? It's Sunny Today."

        With numbers and special characters:

        >>> convert_to_title_case("python3.9_features and camelCaseNames")
        'Python 3.9 Features And Camel Case Names'

    Note:
        - Preserves contractions (e.g., "what's" -> "What's")
        - Handles mixed case formats in the same string
        - Maintains punctuation and spacing
        - Properly capitalizes words after numbers and special characters
    """
    if not isinstance(text, str):
        msg = "Input must be a string"
        raise TypeError(msg)

    # Handle snake_case
    text = text.replace("_", " ")

    # Handle camelCase and PascalCase
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", text)

    # Split the text into words, preserving punctuation
    words = re.findall(r"[\w']+|[.,!?;]", text)

    # Capitalize each word
    result = [word.capitalize() if word.isalpha() or "'" in word else word for word in words]

    # Join the words back together
    return " ".join(result)


def generate_output_filename(
    input_path: str | Path,
    output_path: str | Path,
    dataset_name: str | None = None,
    category: str | None = None,
    mkdir: bool = True,
) -> Path:
    """Generate an output filename based on the input path.

    This function generates an output path that preserves the directory structure after the
    dataset root, using an improved algorithm that works with any folder structure.

    Args:
        input_path (str | Path): Path to the input file.
        output_path (str | Path): Base output directory path.
        dataset_name (str | None, optional): Name of the dataset to find in the input path.
            If provided, the path structure after this dataset directory is preserved.
            If not provided or not found, uses intelligent heuristics.
            Defaults to ``None``.
        category (str | None, optional): Category name to find in the input path after
            dataset name. If provided, preserves structure after this category.
            Defaults to ``None``.
        mkdir (bool, optional): Whether to create the output directory structure.
            Defaults to ``True``.

    Returns:
        Path: Generated output file path preserving relevant directory structure.

    Examples:
        Basic usage with MVTec-style dataset:

        >>> input_path = "/data/MVTecAD/bottle/test/broken_large/000.png"
        >>> generate_output_filename(input_path, "./results", "MVTecAD", "bottle")
        PosixPath('results/test/broken_large/000.png')

        Without category preserves more structure:

        >>> generate_output_filename(input_path, "./results", "MVTecAD")
        PosixPath('results/bottle/test/broken_large/000.png')

        Works with folder datasets:

        >>> path = "/datasets/MyDataset/normal/image001.png"
        >>> generate_output_filename(path, "./output", "MyDataset")
        PosixPath('output/normal/image001.png')

        Handles custom structures gracefully:

        >>> path = "/custom/path/MyData/category/split/image.png"
        >>> generate_output_filename(path, "./out", "MyData")
        PosixPath('out/category/split/image.png')

        Auto-detection when dataset_name not provided:

        >>> path = "/any/folder/structure/normal/image.png"
        >>> generate_output_filename(path, "./out")
        PosixPath('out/normal/image.png')

        Case-insensitive matching:

        >>> path = "/data/mvtecad/BOTTLE/test/000.png"
        >>> generate_output_filename(path, "./results", "MVTecAD", "bottle")
        PosixPath('results/test/000.png')

        Relative paths:

        >>> path = "data/MVTecAD/bottle/test/000.png"
        >>> generate_output_filename(path, "./results", "MVTecAD")
        PosixPath('results/bottle/test/000.png')

    Note:
        - Uses intelligent path analysis to work with any folder structure
        - Preserves directory structure after the dataset root
        - If ``mkdir=True``, creates output directory structure if it doesn't exist
        - Original filename is always preserved in output path
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Find the base path to exclude
    exclude_base = None

    # Try to find dataset_name in path
    if dataset_name:
        for i, part in enumerate(input_path.parts):
            if part.lower() == dataset_name.lower():
                exclude_base = Path(*input_path.parts[: i + 1])
                break

    # Try to find category after dataset
    if exclude_base and category:
        try:
            remaining = input_path.relative_to(exclude_base)
            for j, part in enumerate(remaining.parts):
                if part.lower() == category.lower():
                    exclude_base = exclude_base / Path(*remaining.parts[: j + 1])
                    break
        except ValueError:
            pass  # relative_to failed, keep original exclude_base

    # Use relative_to to get the remaining path structure
    if exclude_base:
        try:
            relative_path = input_path.relative_to(exclude_base)
            preserved_dirs = relative_path.parts[:-1]  # All dirs except filename
        except ValueError:
            # Fallback: keep last directory
            preserved_dirs = (input_path.parent.name,)
    else:
        # No dataset found, keep last directory
        preserved_dirs = (input_path.parent.name,)

    # Build final path
    final_output_path = output_path / Path(*preserved_dirs) if preserved_dirs else output_path

    if mkdir:
        final_output_path.mkdir(parents=True, exist_ok=True)

    return final_output_path / input_path.name
