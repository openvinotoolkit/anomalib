# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Atheris target for paths read from untrusted dataset metadata."""

import os
import sys
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from anomalib.data.utils import path as path_utils

try:
    import atheris
except ImportError:
    atheris = None


def fuzz_path(data: bytes, root: Path) -> None:
    """Check containment, extension validation, and acceptance of valid paths.

    Args:
        data: Mutated metadata path bytes.
        root: Resolved dataset directory, optionally containing an escape symlink.
    """
    value = data.decode("utf-8", errors="surrogateescape")
    if os.name == "nt" and value.replace("/", "\\").startswith("\\\\"):
        # shortcut: avoid remote UNC access on Windows; Linux CI covers these inputs.
        return
    for candidate in (value, str(root / value), str(root.parent / value)):
        for extensions in (None, (".png",)):
            try:
                result = path_utils.resolve_path_under_root(root, candidate, should_exist=False, extensions=extensions)
            except (ValueError, OSError):
                continue
            # Check containment independently of the production helper.
            assert os.path.commonpath([root, result]) == str(root)
            assert result.is_absolute()
            if extensions:
                assert result.suffix in extensions

    # An always-rejecting implementation must fail too.
    relative = Path("images") / f"{data[:32].hex()}.png"
    expected = root / relative
    assert path_utils.resolve_path_under_root(root, relative, should_exist=False, extensions=(".png",)) == expected
    assert path_utils.resolve_path_under_root(root, expected, should_exist=False) == expected
    rejected = [
        Path("..") / relative,
        root.parent / "dataset-sibling" / relative,
    ]
    if (root / "escape").is_symlink():
        rejected.append(root / "escape" / relative)
    for candidate in rejected:
        try:
            path_utils.resolve_path_under_root(root, candidate, should_exist=False)
        except ValueError:
            continue
        msg = f"Accepted an escaping metadata path: {candidate}"
        raise AssertionError(msg)


def main() -> None:
    """Instrument the path boundary and run Atheris with a stable filesystem."""
    if atheris is None:
        msg = "Atheris is required to run fuzzing. Install the fuzz extra on Linux or macOS."
        raise ImportError(msg)

    # Instrument only the boundary, avoiding unrelated ML imports and native code.
    for function in (
        path_utils.resolve_path_under_root,
        path_utils.validate_path,
        path_utils.is_within_directory,
        path_utils.is_path_too_long,
        path_utils.contains_non_printable_characters,
        fuzz_path,
    ):
        atheris.instrument_func(function)
    with TemporaryDirectory(prefix="metadata-fuzz-") as directory:
        root = Path(directory).resolve() / "dataset"
        root.mkdir()
        outside = root.parent / "outside"
        outside.mkdir()
        (root / "escape").symlink_to(outside, target_is_directory=True)
        atheris.Setup(sys.argv, partial(fuzz_path, root=root))
        atheris.Fuzz()


if __name__ == "__main__":
    main()
