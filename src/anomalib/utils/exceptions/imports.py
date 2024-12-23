"""Import handling utilities for anomaly detection.

This module provides utilities for handling dynamic imports and import-related
exceptions in the anomalib library. The utilities include:
    - Dynamic module import with graceful error handling
    - Import availability checking
    - Deprecation warnings for legacy import functions

Example:
    >>> from anomalib.utils.exceptions import try_import
    >>> # Try importing an optional dependency
    >>> torch_fidelity = try_import("torch_fidelity")
    >>> if torch_fidelity is None:
    ...     print("torch-fidelity not installed")

The module ensures consistent handling of optional dependencies and provides
helpful error messages when imports fail.
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from importlib import import_module

logger = logging.getLogger(__name__)


def try_import(import_path: str) -> bool:
    """Try to import a module and return whether the import succeeded.

    This function attempts to dynamically import a Python module and handles any
    import errors gracefully. It is deprecated and will be removed in v2.0.0.
    Users should migrate to ``module_available`` from lightning-utilities instead.

    Args:
        import_path (str): The import path of the module to try importing. This can
            be a top-level package name (e.g. ``"torch"``) or a submodule path
            (e.g. ``"torch.nn"``).

    Returns:
        bool: ``True`` if the import succeeds, ``False`` if an ``ImportError``
            occurs.

    Warns:
        DeprecationWarning: This function is deprecated and will be removed in
            v2.0.0. Use ``module_available`` from lightning-utilities instead.

    Example:
        >>> from anomalib.utils.exceptions import try_import
        >>> # Try importing an optional dependency
        >>> has_torch = try_import("torch")
        >>> if not has_torch:
        ...     print("PyTorch is not installed")
        >>> # Try importing a submodule
        >>> has_torchvision = try_import("torchvision.transforms")
    """
    import warnings

    warnings.warn(
        "The 'try_import' function is deprecated and will be removed in v2.0.0. "
        "Use 'module_available' from lightning-utilities instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    try:
        import_module(import_path)
    except ImportError:
        import_package = import_path.split(".")[0]
        logger.warning(
            f"Could not find {import_package}. To use this feature, ensure that you have {import_package} installed.",
        )
    else:
        return True
    return False
